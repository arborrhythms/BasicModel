"""Reduce all predeclared arms without selecting seeds or omitting bad rows."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

CONTROLS = ('ordered', 'shuffled', 'context_free', 'reconstruction_only')


def mean(rows):
    return statistics.mean(rows) if rows else None


def thought_summary(thought):
    samples = thought['trials']
    result = dict(pairs=thought['pairs'], unsupported=thought['unsupported'], conditions={})
    for gain, rows in samples.items():
        valid = [r for r in rows if 'error' not in r]
        result['conditions'][gain] = dict(answered=len(valid),
            errors=dict(Counter(r['message'] for r in rows if 'error' in r)),
            work=mean([r['work'] for r in valid]), steps=mean([r['steps'] for r in valid]),
            assertion_brier=mean([r['assertion_brier'] for r in valid]))
    pairs = [(a, b) for a, b in zip(samples['0.0'], samples['1.0'])
             if 'error' not in a and 'error' not in b]
    matched = [(a, b) for a, b in pairs if abs(a['assertion_brier']-b['assertion_brier']) <= 1e-6]
    useful = [(a, b) for a, b in matched if a['assertion_brier'] < 1. and b['assertion_brier'] < 1.]
    result.update(matched_error_pairs=len(matched), matched_pairs_better_than_neither=len(useful),
        answer_disagreement=sum(a['answer'] != b['answer'] for a, b in pairs),
        useful_matched_work_delta=mean([b['work']-a['work'] for a, b in useful]),
        demonstrated_work_advantage=bool(useful) and len(pairs) == thought['pairs']
            and mean([b['work']-a['work'] for a,b in useful]) < 0)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs', type=Path, nargs=3)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    sources = []
    report = dict(seeds=[], protocol='PROTOCOL.md', numerical_tolerance=1e-6)
    for seed, directory in enumerate(args.runs):
        arms, finals = {}, {}
        for control in CONTROLS:
            path = directory / f'{control}-{seed}' / 'result.json'
            data = json.loads(path.read_text())
            assert data['seed'] == seed and data['control'] == control
            assert data.get('source_unchanged') and 'error' not in data
            sources.append(data['source'])
            before, training, after = data['phases']
            assert [len(v['steps']) for v in data['phases']] == [16, 64, 16]
            assert sum(s['optimizer_steps'] for v in data['phases'] for s in v['steps']) == 64
            finals[control] = after
            label = control if control != 'reconstruction_only' else 'ordered'
            steps = [s for phase in data['phases'] for s in phase['steps']]
            arms[control] = dict(result=str(path),
                result_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                predictor_delta=data['predictor_delta'],
                current_encoding=after['scores'][label],
                fixed_pretraining_encoding=after['fixed_pretraining_encoding'],
                initial_encoding=before['scores'][label],
                training_context_control=training['context_control'],
                training_reconstruction=training['reconstruction_mean'],
                final_reconstruction=after['reconstruction_mean'],
                final_byte_cost=mean([s['fidelity']['byte_cost'] for s in after['steps']]),
                incomplete_reconstruction_rows=sum(s['fidelity']['truncated_rows'] for s in steps),
                basis_limited_sentences=sum(s['fidelity']['basis_limited_sentences'] for s in steps),
                maximum_sentence_basis=max(s['fidelity']['maximum_sentence_basis'] for s in steps),
                discrimination=data['discrimination']['categorical_discrimination'])
            if control == 'ordered':
                thought = thought_summary(data['thought'])
        # Identical corpus identities and scored target counts are prerequisites
        # for the comparison; do not silently shorten one condition.
        assert all([s['sources'] for s in finals[c]['steps']] ==
                   [s['sources'] for s in finals['ordered']['steps']] for c in CONTROLS)
        gates = {}
        for encoding in ('current_encoding', 'fixed_pretraining_encoding'):
            gates[encoding] = {metric: all(arms['ordered'][encoding][metric] < arms[c][encoding][metric]
                for c in ('shuffled', 'context_free')) for metric in ('feature_mse', 'presence_bce')}
        gates['reconstruction_no_worse'] = arms['ordered']['final_byte_cost'] <= arms['reconstruction_only']['final_byte_cost'] + 1e-6
        gates['discrimination_no_worse'] = {probe: arms['ordered']['discrimination'][probe]['cp'] + 1e-6 >=
            arms['reconstruction_only']['discrimination'][probe]['cp'] for probe in ('xor','fineweb')}
        gates['thought_work_advantage'] = thought['demonstrated_work_advantage']
        report['seeds'].append(dict(seed=seed, arms=arms, thought=thought, gates=gates))
    assert all(source == sources[0] for source in sources)
    report['source'] = sources[0]
    report['source_sha256'] = hashlib.sha256(json.dumps(sources[0], sort_keys=True).encode()).hexdigest()
    def passed(value):
        return all(passed(v) for v in value.values()) if isinstance(value,dict) else bool(value)
    report['all_learning_gates_pass'] = all(passed(seed['gates']) for seed in report['seeds'])
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source','seeds')}, indent=2))
    print(json.dumps([dict(seed=s['seed'], gates=s['gates'], thought=s['thought']) for s in report['seeds']], indent=2))


if __name__ == '__main__':
    main()
