"""Read saved observations only; never construct or rerun a model."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ARMS = {
    'XOR_grammar': 'xor-ownership',
    'BasicModel_answers_tied_benchmark':
        'native-stage1/BasicModel_answers_tied_benchmark-ownership',
}


def stats(values):
    values = [float(v) for v in values if v is not None and math.isfinite(v)]
    return (dict(count=len(values), minimum=min(values), median=statistics.median(values),
                 mean=statistics.mean(values), maximum=max(values)) if values else None)


def read(folder, name):
    path = folder / name
    return json.loads(path.read_text()) if path.exists() else None


def selection_audit(events):
    result = dict(active_rows=0, explore_kept=0, selection_rule_violations=0)
    gaps = defaultdict(list)
    observed = Counter()
    for event in events:
        if event['kind'] != 'selection':
            continue
        for i, active in enumerate(event['active']):
            if not active:
                continue
            win = event['wins'][i]
            result['active_rows'] += 1
            result['explore_kept'] += int(win)
            expected_win = event['explore']['reconstruction'][i] < event['exploit']['reconstruction'][i]
            result['selection_rule_violations'] += int(bool(win) != bool(expected_win))
            kept, rejected = ('explore', 'exploit') if win else ('exploit', 'explore')
            for objective in ('reconstruction', 'expectation', 'supplied_answer', 'total'):
                a, b = event[kept][objective][i], event[rejected][objective][i]
                if a is not None and b is not None:
                    observed[objective] += 1
                    gaps[objective].append(a - b)
    for name in ('reconstruction', 'expectation', 'supplied_answer', 'total'):
        worse = [gap for gap in gaps[name] if gap > 0]
        result[name] = dict(observed=observed[name], worse=len(worse),
                            worse_gaps=stats(worse), all_kept_minus_other=stats(gaps[name]))
    return result


def term_magnitudes(events):
    terms = defaultdict(list)
    for event in events:
        if event['kind'] not in ('trial', 'batch'):
            continue
        location = ('trial.' + event['trial'] if event['kind'] == 'trial'
                    else 'batch_end')
        location += '.train' if event.get('train') else '.eval'
        for name, term in event.get('terms', {}).items():
            terms[(location, name)].append(term)
    rows = []
    for (location, name), values in sorted(terms.items()):
        row = dict(location=location, name=name,
                   category=sorted({v.get('category', '') for v in values}),
                   objective=sorted({v.get('objective', '') for v in values}),
                   kind=sorted({v.get('kind', '') for v in values}),
                   trained=sorted({v.get('trained', False) for v in values}),
                   weights=sorted({v['weight'] for v in values}),
                   context_weights=sorted({v.get('context_weight', 1) for v in values}))
        for key in ('value', 'weighted', 'raw', 'baseline', 'active_entries'):
            row[key] = stats([v.get(key) for v in values])
        rows.append(row)
    return rows


def summarize(folder):
    outcome = read(folder, 'outcome.json')
    ownership = read(folder, 'ownership.json')
    if outcome is None or ownership is None:
        return None
    events = [json.loads(line) for line in (folder / 'events.jsonl').read_text().splitlines()]
    # Coordinate arrays are the primary displacement record. Recompute
    # norms/cosines in float64, retaining the online float32 events unchanged.
    import numpy as np
    for event in events:
        if event['kind'] != 'optimizer_displacement':continue
        with np.load(folder/event['file']) as arrays:
            for row in event['parameters']:
                if not row['gradient_present']:continue
                key=row['key']
                gradient=arrays[key+'_gradient'].astype(np.float64).reshape(-1)
                delta=arrays[key+'_displacement'].astype(np.float64).reshape(-1)
                ng=float(np.linalg.norm(gradient));nd=float(np.linalg.norm(delta))
                row.update(gradient_norm=ng,displacement_norm=nd,
                    displacement_over_gradient=None if ng==0 else nd/ng,
                    cosine=None if ng*nd==0 else float(np.dot(gradient,delta)/(ng*nd)))
    gradients = read(folder, 'gradients.json')
    compact_gradients = []
    for state in gradients:
        row = {key: state.get(key) for key in
               ('scope', 'trial', 'pair', 'batch', 'version_sha256', 'costs', 'rng_unchanged')}
        row['groups'] = {group: {k: data[k] for k in ('norms', 'cosines')}
                         for group, data in state['groups'].items()}
        compact_gradients.append(row)
    ranking = Counter()
    for event in events:
        if event['kind'] == 'activated_word_ranking':
            for key in ('words', 'with_activated_candidate', 'activated_outranks_own'):
                ranking[key] += event[key]
    batches = [e for e in events if e['kind'] == 'batch']
    endpoint = dict(outcome['last_batch'])
    objective_costs = endpoint['objective_costs']
    endpoint['with_answer'] = sum(v for v in objective_costs.values() if v is not None)
    endpoint['without_answer'] = sum(v for k, v in objective_costs.items()
                                     if k != 'output' and v is not None)
    last_training = next((e for e in reversed(batches) if e.get('train')), None)
    last_trials = {}
    for name, trial in outcome['last_training_trials'].items():
        last_trials[name] = {key: stats(values) for key, values in trial['weighted'].items()}
    configuration = read(folder, 'configured-training.json')
    if configuration is None:
        configuration = read(folder, 'plan.json')['effective']['architecture']['training']
    return dict(folder=str(folder.relative_to(HERE)),
                configured_training=configuration,
                training_batches=outcome['training_batches'], evaluation_batches=outcome['evaluation_batches'],
                ownership=ownership, gradients=compact_gradients,
                selection=selection_audit(events), term_magnitudes=term_magnitudes(events),
                endpoint=endpoint, last_training_batch=last_training, last_training_trials=last_trials,
                activated_word_ranking=dict(ranking),
                geometry_start=read(folder, 'geometry-start.json'),
                geometry_end=read(folder, 'geometry-end.json'),
                stability=read(folder, 'derivation-stability.json'),
                displacement_events=[e for e in events if e['kind']=='optimizer_displacement'])


def fmt(value):
    if value is None:
        return 'undefined'
    return f'{value:.7g}' if isinstance(value, (float, int)) else str(value)


def main():
    arms = {name: summarize(HERE / path) for name, path in ARMS.items()}
    (HERE / 'audit-summary.json').write_text(json.dumps(arms, indent=2) + '\n')
    lines = ['# Ownership, costs and geometry', '',
        'Derived from the two saved first measurements. No model was rerun. '
        'The XOR audit reuses class run 1; the native audit reuses the production stage-1 test.', '',
        'Gradient tables show raw graph reach before the owner restriction. Structural groups overlap '
        '(for example a perceptual synthesis module also belongs to generation). The per-parameter '
        'ownership files show the actual permitted/applied writer. A cosine is undefined when either '
        'gradient has zero norm. These zeros do not establish how an inactive objective would behave.', '',
        'Term statistics are over logged event-level means, not a pooled estimator over unequal batches. '
        'Trial and batch totals are separately reported locations; they must not be added again. '
        'The endpoint is the saved evaluation batch after training; the last training batch is also '
        'retained in the JSON. Penalties and reporting/count entries are listed separately from trained '
        'relative errors by their `kind` and `trained` fields.', '',
        'Term definitions, targets, uninformed baselines, purpose and owner are in '
        '[GradientFlow](../../GradientFlow.md). Reconstruction has only the free '
        'byte term, divided by log(256) and weighted by reconstructionScale. '
        'The supplied squared-error answer uses the detached target mean-square norm. '
        'Zero-target squared terms are penalties. concept_readout_l1 retains its own coefficient '
        'as a proximal penalty rather than a normalized objective.', '']
    for name, arm in arms.items():
        lines += ['## ' + name, '']
        if arm is None:
            lines += ['Measurement has not finished.', '']
            continue
        own = arm['ownership']; folder = arm['folder']
        lines += [f"Declared parameter records: {len(own['parameters'])}; active: {own['active']}; "
                  f"inactive: {own['inactive']}; conflicts: **{own['conflicts']}**; "
                  f"training backward calls: {own['backward_steps']}.", '',
                  f'[Every declared and observed writer]({folder}/ownership.json); '
                  f'[complete per-parameter gradients]({folder}/gradients.json); '
                  f'[configuration record]({folder}/' + ('plan.json' if (HERE / folder / 'plan.json').exists()
                                                       else 'configured-training.json') + ').', '',
                  'Configured priorities can be inactive in the observed data. The term table below shows '
                  'what was actually registered, in each location; an absent term is not a measured zero.', '',
                  '| Setting | Value | Purpose |', '|---|---:|---|']
        priorities = {
            'reconstructionScale': 'Weight of the free read-back relative error',
            'whatScale': 'Supplied-answer what band',
            'whereScale': 'Supplied-answer where band, when present',
            'whenScale': 'Supplied-answer when band, when present',
            'intraLossWeight': 'Within-sentence prediction',
            'interLossWeight': 'Between-sentence prediction',
            'interContrastiveWeight': 'Between-sentence contrastive prediction',
            'armaScale': 'ARMA prediction',
            'grammarLessonWeight': 'Annotated compose/generate chooser lessons',
            'embeddingScale': 'Lexical embedding targets, when present',
            'expectationPolicyWeight': 'Retired expectation policy route; stays zero',
            'selectedThoughtPolicyWeight': 'Answer-owned selected-thought policy',
            'TruthLoss': 'Stored-truth penalty strength',
        }
        for key, purpose in priorities.items():
            if key in arm['configured_training']:
                lines.append(f"| `{key}` | {fmt(arm['configured_training'][key])} | {purpose} |")
        lines += ['',
                  '### First-state gradients', '',
                  '| State | Group | Reconstruction norm | Expectation norm | Answer norm | R/E cosine | R/A cosine | E/A cosine |',
                  '|---|---|---:|---:|---:|---:|---:|---:|']
        for state in arm['gradients']:
            label = state['scope'] + ('.' + state['trial'] if state['trial'] else '')
            for group, values in state['groups'].items():
                columns = [values['norms'][o] for o in ('reconstruction', 'expectation', 'supplied_answer')]
                columns += [values['cosines'][o] for o in
                            ('reconstruction__expectation', 'reconstruction__supplied_answer', 'expectation__supplied_answer')]
                lines.append('| ' + label + ' | ' + group + ' | ' + ' | '.join(map(fmt, columns)) + ' |')
        trial_states = [s for s in arm['gradients'] if s['scope'] == 'trial']
        lines += ['', 'First trial parameter hashes: ' + ', '.join('`' + s['version_sha256'] + '`' for s in trial_states) + '.',
                  'RNG unchanged by each gradient observation: ' + str([s['rng_unchanged'] for s in arm['gradients']]) + '.', '',
                  '### Trial selection', '',
                  f"Active training rows: {arm['selection']['active_rows']}; explore kept: {arm['selection']['explore_kept']}; "
                  f"selection-rule violations: {arm['selection']['selection_rule_violations']}.", '',
                  'The kept trial has the lower reconstruction; equality keeps greedy. Answer and expectation '
                  'can be worse because neither enters the comparison.', '',
                  '| Objective | Comparable rows | Kept worse | Mean positive gap | Maximum positive gap |',
                  '|---|---:|---:|---:|---:|']
        for objective in ('reconstruction', 'expectation', 'supplied_answer', 'total'):
            row = arm['selection'][objective]; worse = row['worse_gaps'] or {}
            lines.append(f"| {objective} | {row['observed']} | {row['worse']} | {fmt(worse.get('mean'))} | {fmt(worse.get('maximum'))} |")
        lines += ['', '### Every recorded cost term', '',
                  'Values are relative errors for `relative` rows. Each magnitude column is median [minimum, maximum]. '
                  'Exact means, raw errors, baselines and active-entry counts are in audit-summary.json.', '',
                  '| Location | Term | Kind / trained | Weight | Relative or penalty value | Weighted value | Baseline |',
                  '|---|---|---|---:|---|---|---|']
        def magnitude(data):
            return '—' if data is None else f"{fmt(data['median'])} [{fmt(data['minimum'])}, {fmt(data['maximum'])}]"
        for row in arm['term_magnitudes']:
            lines.append('| ' + ' | '.join([row['location'], '`' + row['name'] + '`',
                         ','.join(row['kind']) + ' / ' + str(row['trained']), str(row['weights']),
                         magnitude(row['value']), magnitude(row['weighted']), magnitude(row['baseline'])]) + ' |')
        end = arm['endpoint']
        lines += ['', 'Endpoint objective costs: `' + json.dumps(end['objective_costs']) + '`.',
                  f"With the answer: {fmt(end['with_answer'])}; without it: {fmt(end['without_answer'])}.", '',
                  'Expectation is evaluated in the training trial, not recomputed in the evaluation forward. '
                  'Its final recorded value is therefore given below with the other final training-trial costs; '
                  '`undefined` at evaluation means absent, not zero.', '',
                  '| Last training trial | R | E | A | Comparison R |',
                  '|---|---:|---:|---:|---:|']
        for name, trial in arm['last_training_trials'].items():
            vals = [(trial[k] or {}).get('mean') for k in
                    ('reconstruction', 'expectation', 'supplied_answer', 'total')]
            lines.append('| ' + name + ' | ' + ' | '.join(map(fmt, vals)) + ' |')
        lines += ['',
                  '### Code geometry and activated competitors', '',
                  '| State | Physical dictionary | Rows × dimension | Norm min / mean / max | Pair cosine mean / mean-square | EMA refresh | Cluster size |',
                  '|---|---:|---|---|---|---|---|']
        for key in ('geometry_start', 'geometry_end'):
            geo = arm[key]
            for d in geo['dictionary']:
                norms = d['norms']; pairs = d['all_dictionary_pairs']
                lines.append('| ' + ' | '.join([key, str(d['stage']), f"{d['rows']} × {d['dimension']}",
                    ' / '.join(fmt(norms[k]) for k in ('min', 'mean', 'max')),
                    ' / '.join(fmt(pairs[k]) for k in ('mean', 'mean_square')), str(d['vq_ema_update']),
                    str(d.get('cluster_size'))]) + ' |')
        for key in ('geometry_start', 'geometry_end'):
            roots = arm[key]['roots']
            lines += ['', f"{key} roots: shape {roots['shape']}; singular values {roots['singular_values']}; "
                      f"centered singular values {roots['centered_singular_values']}.", '']
        if arm['geometry_start']['roots']['shape'] != arm['geometry_end']['roots']['shape']:
            lines += ['The native root samples are different training/evaluation batches (28 and 16 rows); '
                      'their spectra are not a before/after comparison of identical sentences. XOR uses '
                      'the same four sentences at both endpoints.', '']
        lines += ['', '### Gradient displacement and derivation stability', '',
                  'Every observed optimizer step has coordinate arrays under `displacements/`; '
                  'events map each array to its parameter and sparse row ids. The table aggregates '
                  'step norms and cosines recomputed in float64 from the arrays, not individual coordinates. Momentum can make a later '
                  'step differ from the negative current gradient.', '',
                  '| Parameter | Steps with gradient | Gradient norm median | Displacement norm median | Displacement / gradient median | Cosine median |',
                  '|---|---:|---:|---:|---:|---:|']
        displacement = defaultdict(list)
        for event in arm['displacement_events']:
            for row in event['parameters']:
                if row['gradient_present']:
                    displacement[row['parameter']].append(row)
        for parameter, values in displacement.items():
            medians = []
            for key in ('gradient_norm','displacement_norm','displacement_over_gradient','cosine'):
                summary = stats([v.get(key) for v in values])
                medians.append(None if summary is None else summary['median'])
            lines.append('| `'+parameter+'` | '+str(len(values))+' | '+' | '.join(map(fmt,medians))+' |')
        stability = arm['stability'] or []
        lines += ['', '| Sentence word rows | Observed epochs | Modal fraction | Distinct derivations |',
                  '|---|---:|---:|---:|']
        for row in stability:
            lines.append('| '+str(row['word_rows'])+' | '+str(row['epochs'])+' | '+fmt(row['modal_fraction'])+' | '+str(row['distinct_derivations'])+' |')
        lines += ['', 'Native training has one configured epoch. Its modal fraction is descriptive, '
                  'not evidence of stability across repeated epochs. If a native sentence repeats '
                  'within an epoch, its last kept derivation is that epoch’s sample; all trial events remain saved.', '']
        rank = arm['activated_word_ranking']
        lines += [f"Across recorded training trials and evaluation: {rank.get('words', 0)} word occurrences, "
                  f"{rank.get('with_activated_candidate', 0)} with an activated candidate, "
                  f"{rank.get('activated_outranks_own', 0)} outranked by an activated candidate. "
                  'A zero candidate count supplies no evidence about competition with activated words.', '',
                  f'Full pairwise tensors and row IDs are beside [{folder}/geometry-end.json]({folder}/geometry-end.json). '
                  'Native reserve-wide pair moments cover all row pairs exactly; full matrices cover all observed '
                  'primed rows. A missing cluster-size buffer in the indexed-only native allocation is reported '
                  'as absent, not as a measured vector of zeroes.', '']
    (HERE / 'audits.md').write_text('\n'.join(lines))
    print(json.dumps({name: None if arm is None else {'conflicts': arm['ownership']['conflicts'],
        'selection': arm['selection'], 'ranking': arm['activated_word_ranking']} for name, arm in arms.items()}))


if __name__ == '__main__':
    main()
