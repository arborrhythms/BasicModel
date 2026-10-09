"""Summarize every declared attempt, keeping failure separate from accuracy."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def main():
    protocol = read(HERE / 'protocol.json')
    assert read(HERE / 'math-trainings/complete.json', {}).get('attempts') == 30
    assert read(HERE / 'measurements/complete.json', {}).get('completed') is True
    result = dict(protocol=protocol, source=read(HERE / 'measured-source/freeze.json'),
                  conditions={}, no_retry=True, learning_claim=False)
    for name in ('final-sweep', 'thinking-gate'):
        gate = read(HERE / name / 'result.json', {})
        result[name] = dict(reason=gate.get('reason'), exit_code=gate.get('exit_code'),
            selected=len(gate.get('selected', ())), completed=len(gate.get('completed', ())))
    for condition in protocol['conditions']:
        entries = []
        for run in range(1, 11):
            folder = HERE / 'math-trainings' / f'{condition}-{run:02}'
            value = read(folder / 'result.json', {})
            question_file = folder / 'questions.jsonl'
            questions = ([json.loads(line) for line in question_file.read_text().splitlines()]
                         if question_file.exists() else [])
            observer = read(folder / 'observer.json', {})
            entries.append(dict(run=run, process=read(folder / 'process.json'),
                status=value.get('status', 'not_completed'), error=value.get('error'),
                completed_epochs=value.get('completed_epochs', 0),
                held_out=value.get('test'), beyond=value.get('beyond'),
                held_out_bar=value.get('held_out_bar', False),
                chain_bar=value.get('chain_bar', False),
                chooser_movement=value.get('chooser_movement'), observer=observer,
                question_observations=len(questions),
                question_episodes=sum(row['episode'] for row in questions),
                completed_chains=sum(row['chain_correct'] for row in questions),
                seconds=value.get('seconds')))
        tested = sum(len(row['held_out']['rows']) for row in entries if row['held_out'])
        correct = sum(row['held_out']['correct'] for row in entries if row['held_out'])
        result['conditions'][condition] = dict(runs=entries,
            completed=sum(row['status'] == 'completed' for row in entries),
            all_four_correct=sum(row['held_out_bar'] for row in entries),
            held_out_rows_tested=tested, held_out_rows_correct=correct,
            held_out_accuracy=correct/tested if tested else None,
            held_out_chain_runs=sum(row['chain_bar'] for row in entries))
    standing = read(HERE / 'measurements/complete.json', {})
    result['standing'] = dict(completed=len(standing.get('jobs', ())),
        processes_passed=sum(row['process']['exit_code'] == 0 for row in standing.get('jobs', ())))
    normal = result['conditions']['answer_and_expectation']
    result['learning_claim'] = (normal['completed'] == 10 and normal['all_four_correct'] >= 9
                                and normal['held_out_chain_runs'] == 10)
    result['status'] = 'ready_for_review' if result['learning_claim'] else 'learning_gate_not_met'
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    files = [path for path in HERE.rglob('*') if path.is_file()
             and '__pycache__' not in path.parts and path.name != 'receipt-sha256.json']
    hashes = {str(path.relative_to(HERE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    (HERE / 'receipt-sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], conditions={name:
        {key: value for key, value in report.items() if key != 'runs'}
        for name, report in result['conditions'].items()})))


if __name__ == '__main__':
    main()
