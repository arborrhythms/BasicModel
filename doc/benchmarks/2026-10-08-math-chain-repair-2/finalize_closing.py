"""Read-only validation of retained evidence; write the separate closing summary."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'test'))
from bounded_tests import source_snapshot


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def verify(base, manifest):
    files = read(manifest)
    changed = [name for name, value in files.items()
               if not (base/name).is_file() or digest(base/name) != value]
    assert not changed, (str(manifest), changed)
    return dict(files=len(files), unchanged=True, manifest=str(manifest.relative_to(ROOT)))


def gate(kind, revision, source):
    folder = HERE/f'closing-{kind}-{revision}'
    end = read(HERE/f'closing-{kind}-{revision}-exit.json')
    assert end == dict(exit_code=0, source_verified=True, retained_verified=True), end
    assert read(folder/'source-manifest.json')['validated_source'] == source
    result = read(folder/'result.json')
    assert result['exit_code'] == 0 and set(result['completed']) == set(result['selected'])
    cases = {}
    for worker in result['workers']:
        for report in worker['reports']:
            if report['phase'] == 'call' or report['outcome'] != 'passed':
                cases[report['nodeid']] = report['outcome']
    return dict(selected=len(result['selected']), completed=len(result['completed']),
                outcomes=dict(Counter(cases.values())), seconds=result['elapsed_seconds'],
                peak_aggregate_memory_bytes=result['peak_aggregate_memory_bytes'],
                result=str((folder/'result.json').relative_to(HERE)))


def main(source_revision, certificate_revision, sweep_revision, thinking_revision):
    source_folder = HERE/f'closing-source-{source_revision}'
    source = read(source_folder/'source.json')
    assert source_snapshot(ROOT) == source
    integrity = dict(
        closing_helpers=verify(ROOT, source_folder/'helpers.json'),
        stopped_files=verify(HERE, HERE/'stopped-by-decision/retained-files-sha256.json'),
        stop_receipt=verify(HERE/'stopped-by-decision', HERE/'stopped-by-decision/receipt-sha256.json'),
        campaign_helpers=verify(ROOT, HERE/'measured-source/measurement-helpers.json'))
    integrity['prior_receipts'] = [verify(HERE.parent/name, HERE.parent/name/'receipt-sha256.json')
        for name in ('2026-10-07-math-chain', '2026-10-08-math-chain-repair')]
    contracts = read(HERE/'frozen-contracts.json')
    assert all(name == 'bin/BindingAnswers.py' or digest(ROOT/name) == value
               for name, value in contracts.items())
    binding = read(HERE/'binding-answers-verifier.json')
    assert digest(ROOT/'bin/BindingAnswers.py') == binding['current_sha256']
    assert binding['functions_byte_identical']['matches']
    integrity.update(frozen_contracts_unchanged=True, binding_matches_unchanged=True)
    certificates = []
    assert read(HERE/f'closing-certificates-{certificate_revision}-exit.json')['exit_code'] == 0
    for folder in sorted((HERE/f'closing-certificates-{certificate_revision}').glob('*/decomposition-*')):
        if folder.parent.is_symlink():
            continue
        certificate = read(folder/'certificate.json')
        questions = [json.loads(line) for line in (folder/'questions.jsonl').read_text().splitlines()]
        proved = [row for row in questions if row['chain_correct']]
        assert proved and all(row['bound_correct'] for row in proved)
        assert all(episode['state_diff']['unexpected'] == [] for episode in certificate['episodes'])
        certificates.append(dict(successors=certificate['successors'], forced=True,
            attention_budget=certificate['attention_budget'], premise_budget=certificate['premise_budget'],
            questions=questions, episodes=[dict(row=episode['row'], work=episode['work'],
                state_diff=episode['state_diff'],
                operations=[record['operation'] for record in episode['trace']])
                for episode in certificate['episodes']],
            compose_departures=[row['departure'] for row in certificate['rows']],
            source=str((folder/'certificate.json').relative_to(HERE))))
    assert {item['successors'] for item in certificates} == {1, 2}
    sweep = gate('sweep', sweep_revision, source)
    thinking = gate('thinking', thinking_revision, source)
    assert thinking['completed'] == 57 and thinking['outcomes'] == {'passed': 57}
    stopped = read(HERE/'stopped-by-decision/summary.json')
    unforced = [dict(condition=run['condition'], completed_epochs=run['completed_epochs'],
        first_epoch=run['first_epoch'], greedy_what=run['greedy_what'])
        for run in stopped['observations'] if run.get('completed_epochs', 0)]
    result = dict(status='stopped_for_Claude_review', learning_claim=False, committed=False,
        generated_utc=datetime.now(timezone.utc).isoformat(),
        source=read(source_folder/'freeze.json'), integrity=integrity,
        campaign=dict(started=2, declared=30, completed_trainings=0,
            completed_epochs=stopped['completed_epochs'], retries=0, replacements=0,
            held_out_evaluation=None, partials='stopped-by-decision/summary.json'),
        unforced=unforced, forced_certificates=certificates, sweep=sweep, thinking=thinking,
        standing_thirty=dict(reused=True, rerun=False, report='standing-thirty-verification.json',
            mm_bisection='mm-bisection/result.json'),
        limitations=['Forced mechanism demonstration; no learned policy or accuracy claim.',
            'Two successors use explicit nested premises, not learned plus-two recursion.',
            'Question fixture work budgets 512/768; campaign budget 32 was not changed.',
            'Every companion-row outcome is included; compose alternatives can change a premise.'])
    (HERE/'closing-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(status=result['status'], sweep=sweep, thinking=thinking,
        certificates=[dict(successors=item['successors'],
            correct=sum(row['chain_correct'] for row in item['questions']),
            questions=len(item['questions'])) for item in certificates], integrity=integrity)), flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
