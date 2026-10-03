"""Render all observations; never choose a better run or replace a failed arm."""
from collections import Counter, defaultdict
import itertools
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
CONFIGS = ('XOR_grammar', 'BasicModel_answers_tied_benchmark')
OBJECTIVES = ('reconstruction', 'expectation', 'supplied_answer')


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def flatten(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def stats(values):
    data = sorted(x for x in values if math.isfinite(x))
    if not data:
        return dict(count=0)
    def quantile(p):
        j=(len(data)-1)*p
        lo, hi = math.floor(j), math.ceil(j)
        return data[lo]+(data[hi]-data[lo])*(j-lo)
    return dict(count=len(data), minimum=data[0], p10=quantile(.1), median=statistics.median(data),
                mean=statistics.mean(data), p90=quantile(.9), maximum=data[-1])


def number(value):
    return '—' if value is None else f'{value:.7g}'


all_results = {}
lines = ['# Objective-conflicts stage 1 — measurement only', '',
    'One fresh unseeded run per configuration and arm. The cut arm omits the trial answer term only; '
    'batch-end answer training remains. Initializations differ, so these are measured outcomes, '
    'not a controlled estimate of the causal effect of the cut. No stage-2 or item-6.85 design is implemented.', '',
    'Parameter groups are listed by physical parameter identity in each arm’s manifest. They may overlap '
    '(the native reading map includes generation); overlaps are saved explicitly. A zero gradient and an '
    'untrainable or non-parameter codebook are different states. Undefined cosines are shown as —.', '']
lines += ['The XOR tables use `gradients-by-role.json`: the original observer followed '
    'InputSpace’s registered OutputSpace back-reference, so head parameters appeared in perception '
    'and vocabulary parameters appeared in the reading map. `regroup_xor.py` corrects only these '
    'labels and assigns the intra-sentence predictor to expectation. It proves every recovered '
    'nonzero norm/cosine has exactly the same saved parameter support. The original files remain; '
    'no model was rerun. The native observer classifies those owners directly and also includes '
    'the model’s named shared transforms in operators/tied inverses.', '',
    'Both native arms hit the unchanged 8 GiB worker guard before their first cost or gradient '
    'snapshot. Their reach, selection, magnitude and endpoint measurements are unavailable. '
    'The sampled peaks can exceed 8 GiB between guard polls; the limit was not raised.', '']

for config in CONFIGS:
    all_results[config] = {}
    lines += [f'## {config}', '']
    for arm in ('step5a','cut'):
        folder = HERE / (config+'-'+arm)
        process = read(folder/'process.json', {})
        outcome = read(folder/'outcome.json')
        events = [json.loads(s) for s in (folder/'events.jsonl').read_text().splitlines()] if (folder/'events.jsonl').exists() else []
        result = dict(process=process, error=read(folder/'error.json'), completed=bool(read(folder/'complete.json')))
        all_results[config][arm] = result
        lines += [f'### {arm}', '', f"Process: {process.get('reason','not run')}; exit {process.get('exit_code','—')}; "
                  f"{number(process.get('elapsed_seconds'))} seconds; peak "
                  f"{number(process.get('peak_memory_bytes',0)/2**30)} GiB.", '']
        if result['error']:
            lines += [f"Observer/run error: `{result['error']}`", '']
        # Endpoint values are from the existing final evaluation, after all
        # updates. Earlier training costs stay separately labelled below.
        endpoint = [e for e in events if e['kind']=='trial' and not e['train']
                    and e.get('scope')=='batch' and e['batch']==(outcome or {}).get('training_batches')]
        endcost = defaultdict(list)
        for e in endpoint:
            for i,active in enumerate(e['active']):
                if not active:
                    continue
                for objective in OBJECTIVES:
                    value = (e['evaluation_supplied_answer'][i] if objective=='supplied_answer'
                             else e['weighted'][objective][i])
                    if value is not None:
                        endcost[objective].append(value)
                endcost['raw_reconstruction'].append(e['raw']['reconstruction'][i])
        result['endpoint_objectives']={k:stats(v) for k,v in endcost.items()}
        result['last_batch']=(outcome or {}).get('last_batch')
        result['last_training_trials']=(outcome or {}).get('last_training_trials')
        weights=read(folder/'weights.json',{})
        lines += ['Final evaluation trial costs (answer evaluated at the same state in both arms; '
                  'no optimizer step and no change to the kept trial):', '',
                  '| Objective | Rows | Mean | Median | Min | Max |',
                  '|---|---:|---:|---:|---:|---:|']
        for key,value in result['endpoint_objectives'].items():
            lines.append('| '+' | '.join([key,str(value['count']),*(number(value.get(k)) for k in ('mean','median','minimum','maximum'))])+' |')
        if not endpoint:
            lines += ['| unavailable | 0 | — | — | — | — |']
        lines += ['', 'The endpoint reconstruction is the trial’s configured objective; '
                  'XOR’s separate batch D3 cost is in `last_batch` and the term table. '
                  'Absent objectives have no fabricated value.', '']
        if outcome:
            lines += ['Last training comparison, before that sentence’s two optimizer updates '
                      '(weighted means over active rows; distinct from the final evaluation above):', '',
                      '| Trial | R | E | Supplied answer | Total |', '|---|---:|---:|---:|---:|']
            for trial,row in outcome['last_training_trials'].items():
                values=[]
                for key in (*OBJECTIVES,'total'):
                    observed=[v for i,v in enumerate(row['weighted'][key]) if row['active'][i] and v is not None]
                    values.append(number(statistics.mean(observed) if observed else None))
                lines.append('| '+trial+' | '+' | '.join(values)+' |')
            if config=='XOR_grammar':
                lines += ['', 'XOR’s legacy intra-sentence expectation is evaluated only during '
                          'training. Its final-evaluation E=0 means that branch is inactive, '
                          'not that prediction is perfect; the last training values above preserve '
                          'the actual expectation comparison. The final batch D3 reconstruction '
                          f"is {outcome['last_batch']['raw']['lossIn']:.9g}; it also appears as lossRev, "
                          'each with the configured reconstruction weight.', '']
        if outcome and config=='XOR_grammar':
            predictions=list(flatten(outcome['predictions']));targets=list(flatten(outcome['targets']))
            readings=outcome['grammar_reconstructions'][-1:]
            recon=readings[0] if readings else {}
            inputs=outcome['inputs']
            exact=sum(Counter(a.split())==Counter((b or '').replace(chr(0),' ').split())
                      for a,b in zip(inputs,recon.get('texts',())))
            quality=dict(answers=predictions,targets=targets,
                correct=sum((a>.5)==(b>.5) for a,b in zip(predictions,targets)),
                mse=statistics.mean((a-b)**2 for a,b in zip(predictions,targets)),
                read_backs=recon.get('texts'),unavailable=recon.get('unavailable'),reconstructed=exact)
            quality['class_bar']=quality['correct']==4 and quality['mse']<.05
            quality['reconstruction_bar']=exact==4 and not any(recon.get('unavailable',[True]))
            result['xor']=quality
            lines += [f"Answers: `{predictions}`; MSE **{quality['mse']:.8g}**; correct **{quality['correct']}/4**; "
                f"class bar **{quality['class_bar']}**; reconstructed **{exact}/4**.",
                f"Read-backs: `{quality['read_backs']}`; unavailable: `{quality['unavailable']}`.", '']
        if arm!='step5a':
            continue
        # Audit every eligible comparison, including ties and exact zeros.
        deltas=defaultdict(list)
        selections=[e for e in events if e['kind']=='selection']
        for e in selections:
            for i,active in enumerate(e['active']):
                if not active:continue
                kept,other=('explore','exploit') if e['wins'][i] else ('exploit','explore')
                for key in OBJECTIVES:
                    a,b=e[kept][key][i],e[other][key][i]
                    if a is not None and b is not None:deltas[key].append(a-b)
        audit={key:dict(comparisons=len(v),worse=sum(x>0 for x in v),
            equal=sum(x==0 for x in v),better=sum(x<0 for x in v),
            signed_delta=stats(v),positive_delta=stats(x for x in v if x>0)) for key,v in deltas.items()}
        result['selection_audit']=audit
        lines += ['Kept minus other trial, in the objective’s weighted trial units. Positive means worse. '
                  'Every active comparison is included; no tolerance discards a conflict.', '',
                  '| Objective | Compared | Kept worse | Mean worsening | Max worsening | Mean signed difference |',
                  '|---|---:|---:|---:|---:|---:|']
        for key,a in audit.items():
            lines.append('| '+' | '.join([key,str(a['comparisons']),str(a['worse']),
                number(a['positive_delta'].get('mean')),number(a['positive_delta'].get('maximum')),
                number(a['signed_delta'].get('mean'))])+' |')
        lines += ['', 'Gradient snapshots use autograd reads with the exact cached-perception pullback. '
                  'Both trials precede their first optimizer step and carry matching parameter-version digests. '
                  'The first later nonzero expectation pair is also retained if the first pair has no expectation.', '']
        gradients=read(folder/'gradients-by-role.json',read(folder/'gradients.json',[]))
        result['gradient_snapshots']=gradients
        reach=defaultdict(lambda:defaultdict(set))
        for g in gradients:
            for name,entry in g['groups'].items():
                for objective,norm in entry['norms'].items():
                    observed=reach[g['scope']][objective]
                    if norm>0:
                        observed.add(name)
        result['sampled_nonzero_reach']={scope:{o:sorted(names) for o,names in objectives.items()}
                                       for scope,objectives in reach.items()}
        lines += ['Observed nonzero reach at the saved states (absence means not observed at these states):', '',
                  '| Scope | Objective | Groups with nonzero gradient |','|---|---|---|']
        for scope,objectives in result['sampled_nonzero_reach'].items():
            for objective,names in objectives.items():
                lines.append('| '+scope+' | '+objective+' | '+(', '.join(names) or 'none observed')+' |')
        lines += ['', f"[Codebook parameter/buffer ownership]({folder.name}/codebook-ownership.json) "
                  'records contextual rotation separately from autograd.', '']
        for g in gradients:
            if g['scope']=='batch_auxiliary':
                lines += [f"Additional batch objectives: `{g['costs']}`. Their full reach and cosines are in `gradients.json`.", '']
                continue
            lines += [f"{g['scope']}, pair {g['pair']}, trial {g['trial']}, parameter digest `{g['version_sha256'][:16]}`:", '',
                '| Group | ‖R‖ | ‖E‖ | ‖A‖ | cos(R,E) | cos(R,A) | cos(E,A) |',
                '|---|---:|---:|---:|---:|---:|---:|']
            for name,v in g['groups'].items():
                lines.append('| '+' | '.join([name,*(number(v['norms'].get(k)) for k in OBJECTIVES),
                    *(number(v['cosines'].get(a+'__'+b)) for a,b in itertools.combinations(OBJECTIVES,2))])+' |')
            lines += ['']
        magnitudes=defaultdict(list)
        def add(prefix,mapping):
            for key,value in mapping.items():magnitudes[prefix+'.'+key].extend(flatten(value))
        for e in events:
            if e['kind']=='trial' and e['train']:
                for section in ('raw','weighted','grammar_terms'):
                    add('trial.'+e['trial']+'.'+section,
                        {k:[v[i] for i,a in enumerate(e['active']) if a] for k,v in e[section].items()})
                intra_weight=weights.get('intra_loss_weight',0)
                if intra_weight and 'legacy_intra' in e['weighted']:
                    add('trial.'+e['trial']+'.raw',dict(legacy_intra=[v/intra_weight
                        for i,v in enumerate(e['weighted']['legacy_intra']) if e['active'][i]]))
            elif e['kind']=='expectation_terms' and e['train']:
                for section in ('raw','weighted'):
                    add('trial.'+str(e['trial'])+'.expectation_components.'+section,e[section])
            elif e['kind']=='batch' and e['train']:
                for section in ('raw','weighted','truth'):add('batch.'+section,e[section])
                magnitudes['batch.accounting_residual'].append(e['accounting_residual'])
                for a in e['auxiliary']:
                    magnitudes['batch.auxiliary.raw.'+a['name']].append(a['raw'])
                    magnitudes['batch.auxiliary.weighted.'+a['name']].append(a['raw']*a['weight'] if a['enabled'] else 0)
                for band in e['bands']:
                    for name,v in band['parts'].items():
                        for mode in ('raw','weighted'):
                            magnitudes['band.'+str(band['trial'])+'.'+band['caller']+'.'+name+'.'+mode].append(v[mode])
        result['term_magnitudes']={k:stats(v) for k,v in sorted(magnitudes.items())}
        table=['# Every observed training cost term', '',
               'Raw and weighted units are separate; zeros remain. Missing terms have count 0. '
               'Per-trial distributions count active rows; batch distributions count batches.', '',
               '| Term | N | Min | p10 | Median | Mean | p90 | Max |',
               '|---|---:|---:|---:|---:|---:|---:|---:|']
        for name,s in result['term_magnitudes'].items():
            table.append('| '+' | '.join([name,str(s['count']),*(number(s.get(k)) for k in ('minimum','p10','median','mean','p90','maximum'))])+' |')
        (folder/'term-magnitudes.md').write_text('\n'.join(table)+'\n')
        lines += [f"[Every cost term’s magnitude]({folder.name}/term-magnitudes.md), "
                  f"[weights and formulas]({folder.name}/weights.json), "
                  f"[parameter membership]({folder.name}/{'parameter-groups-by-role.json' if (folder/'parameter-groups-by-role.json').exists() else 'parameter-groups.json'}).", '']

(HERE/'summary.json').write_text(json.dumps(all_results,indent=2)+'\n')
(HERE/'README.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({config:{arm:dict(completed=value['completed'],reason=value['process'].get('reason'))
                         for arm,value in arms.items()} for config,arms in all_results.items()},indent=2))
