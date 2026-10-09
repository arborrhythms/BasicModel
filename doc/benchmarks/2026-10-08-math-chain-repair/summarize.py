"""Read-only reduction of the frozen protocol's retained outputs."""
from collections import Counter
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def sweep(folder):
    result = read(folder/'result.json')
    if result is None:
        return None
    reports = [report for path in folder.glob('worker-*.json')
               if '.request.' not in path.name
               for report in read(path, {}).get('reports', ())]
    outcomes = Counter(report['outcome'] for report in reports
                       if report['phase'] == 'call' or report['outcome'] == 'skipped')
    return dict(exit_code=result['exit_code'], reason=result['reason'],
        selected=len(result['selected']), completed=len(result['completed']),
        outcomes=dict(outcomes), seconds=result.get('elapsed_seconds'),
        failures=[report for report in reports if report['outcome'] == 'failed'])


def math_run(condition, number):
    folder = HERE/'math-trainings'/f'{condition}-{number:02}'
    result = read(folder/'result.json', {})
    process = read(folder/'process.json', {})
    questions = []
    log = folder/'questions.jsonl'
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                questions.append(json.loads(line))
    first = [q for q in questions if q.get('phase') == 'train' and q.get('epoch') == 1]
    epochs = result.get('completed_epochs', 0)
    return dict(condition=condition, run=number, status=result.get('status', 'no_final_result'),
        completed_epochs=epochs, process_exit=process.get('exit_code'),
        process_reason=process.get('reason'), error=result.get('error'),
        held_out=result.get('test'), beyond_range=result.get('beyond'),
        held_out_bar=result.get('held_out_bar'), chain_bar=result.get('chain_bar'),
        chooser_movement=result.get('chooser_movement'),
        first_epoch=dict(complete=epochs >= 1, questions_observed=len(first),
            questions_still_open_after_episode=sum(bool(q.get('open')) for q in first),
            open_slots_remaining=sum(len(q.get('open') or ()) for q in first),
            episodes_opened=sum(q['episode'] for q in first),
            bindings_correct=sum(q['bound_correct'] for q in first)),
        observer=result.get('observer', read(folder/'observer.json')))


def main():
    protocol = read(HERE/'protocol.json')
    runs = [math_run(condition, number) for condition in protocol['conditions']
            for number in range(1, 11)]
    conditions = {}
    for condition in protocol['conditions']:
        selected = [run for run in runs if run['condition'] == condition]
        first = {key: sum(run['first_epoch'][key] for run in selected) for key in
                 ('questions_observed', 'questions_still_open_after_episode',
                  'open_slots_remaining', 'episodes_opened', 'bindings_correct')}
        conditions[condition] = dict(attempts=sum(run['process_exit'] is not None for run in selected),
            completed=sum(run['status'] == 'completed' for run in selected),
            held_out_evaluations=sum(run['held_out'] is not None for run in selected),
            all_four_held_out=(sum(run['held_out_bar'] is True for run in selected)
                              if any(run['held_out'] is not None for run in selected) else None),
            all_four_chains=(sum(run['chain_bar'] is True for run in selected)
                             if any(run['held_out'] is not None for run in selected) else None),
            complete_first_epochs=sum(run['first_epoch']['complete'] for run in selected),
            first_epoch=first)
    result = dict(status='measurement outputs; acceptance requires review', seed=None, retries=0,
        sweep=sweep(HERE/'final-sweep'), thinking_gate=sweep(HERE/'thinking-gate'),
        standing_campaign=read(HERE/'measurements/complete.json'),
        math_campaign=read(HERE/'math-trainings/complete.json'),
        conditions=conditions, math_runs=runs,
        open_reference_measurement='The frozen observer records remaining open slots after the episode, '
            'and whether an episode ran; it does not record the binder\'s initial slot count separately.',
        missing_evaluation='An absent evaluation is unavailable, never scored as zero accuracy.',
        prior_receipt=read(HERE/'integrity-before-measurement.json'))
    (HERE/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(conditions=conditions,
        sweep=None if result['sweep'] is None else result['sweep']['reason'],
        thinking_gate=None if result['thinking_gate'] is None else result['thinking_gate']['reason'])))


if __name__ == '__main__':
    main()
