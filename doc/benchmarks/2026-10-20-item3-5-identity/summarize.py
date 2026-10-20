"""Render measurements already in this receipt; never execute or regrade a run."""
import json
from collections import Counter
from pathlib import Path

receipt = Path(__file__).resolve().parent


def value(x):
    return '`' + json.dumps(x, ensure_ascii=False).replace('|', r'\|') + '`'


def question_table(title, data, source):
    text = [f'## {title}', '', '| Measurement | Result |', '|---|---|']
    text += [f'| {key} | {value(item)} |' for key, item in data.items()]
    text += ['', f'Diagnostics: [{source.name}]({source.relative_to(receipt)}).', '']
    return text


def runs():
    rows, failures = [], {}
    for source in sorted(receipt.glob('*/result.json')):
        report = json.loads(source.read_text())
        if report['reason'] == 'running':
            continue
        outcomes = {}
        failed = []
        for worker in report.get('workers', ()):
            for case in worker.get('reports', ()):
                if case['phase'] == 'call' or case['outcome'] in ('failed', 'skipped'):
                    if outcomes.get(case['nodeid']) != 'failed':
                        outcomes[case['nodeid']] = case['outcome']
                if case['outcome'] == 'failed':
                    failed.append(dict(nodeid=case['nodeid'], phase=case['phase'],
                                       message=case.get('message', '')))
        count = Counter(outcomes.values())
        rows.append([source.parent.name, len(report['completed']),
                     len(report['selected']), count['passed'], count['failed'],
                     count['skipped'], count['xfailed'], count['xpassed'],
                     round(report['elapsed_seconds'], 2), report['reason']])
        if failed:
            failures[source.parent.name] = failed
    lines = ['# Completed runs', '',
             'One case is counted once. Passing teardown phases do not hide failed calls.',
             'A timeout, incomplete run, skip or XPASS is not counted as a pass.', '',
             '| Run | Completed / selected | Pass | Fail | Skip | XFAIL | XPASS | Seconds | Result |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name, done, total, passed, failed, skipped, xfail, xpass, seconds, result in rows:
        lines.append(f'| [{name}]({name}/result.json) | {done}/{total} | {passed} | {failed} | '
                     f'{skipped} | {xfail} | {xpass} | {seconds} | {result} |')
    (receipt/'runs.md').write_text('\n'.join(lines)+'\n')
    (receipt/'failed-cases.json').write_text(json.dumps(failures, indent=2)+'\n')


def generated(suffix):
    lines = ['# G1–G3: one table per generated question', '',
             'These are the unchanged generators and assertions on the new scorer.', '']
    for gate in ('G1', 'G2', 'G3'):
        for source in sorted((receipt/f'generated-{suffix}'/gate).glob('*.json')):
            data = json.loads(source.read_text())
            lines += question_table(f'{gate}: {data["case"]}', data, source)
    (receipt/'generated-questions.md').write_text('\n'.join(lines))


def decomposition(suffix):
    lines = ['# Native decomposition: one table per companion question', '',
             'Both original chain lengths, both gain arms, original measurement seeds.',
             'All question rows are reported; the certificate assertions are unchanged.', '']
    for arm in ('zero', 'enabled'):
        folder = receipt/f'bind-decomposition-{arm}-{suffix}-reports'
        for source in sorted(folder.glob('*/decomposition-*/questions.jsonl')):
            chain = source.parent.name.rsplit('-', 1)[-1]
            for raw in source.read_text().splitlines():
                record = json.loads(raw)
                data = {key: record[key] for key in (
                    'pair', 'answer', 'bound_correct', 'expected_chain_length',
                    'inference_count', 'chain_correct', 'work', 'open', 'committed_binding')}
                lines += question_table(f'{arm}: chain {chain}, document {record["document"]}',
                                        data, source)
    (receipt/f'decomposition-{suffix}-questions.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--decomposition', default='repair')
    parser.add_argument('--generated', default='08')
    args = parser.parse_args()
    runs()
    generated(args.generated)
    decomposition(args.decomposition)
