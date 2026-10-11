"""Summarize completed identity branches; never present text or change a model."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'test'))
from identity_measurement import summarize, valid_reference


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = ROOT / 'output/item6-identity-meaning'
    check = read(run / 'source-check.json')
    status = read(run / 'branch-status.json')
    warm = digest(run / 'warmup/warmup.ckpt')
    reports = {}
    for name in read(run / 'plan.json')['branches']:
        if not name.startswith('identity-'):
            continue
        folder = run / name
        start = read(folder / 'start.json')
        report = dict(process_exit_code=status.get(name), start=start,
                      shared_warmup=start['checkpoint_sha256'] == warm)
        if not (folder / 'results.json').exists():
            report['completed'] = False
            report['failure'] = (read(folder / 'failure.json')
                                 if (folder / 'failure.json').exists() else None)
            reports[name] = report
            continue
        native = read(folder / 'results.json')
        rows = native['results']
        assert native['summary'] == summarize(rows), 'summary differs from raw rows'
        bindings = [r for r in rows if r.get('binding')]
        present = [p for r in bindings for p in r['native_bank']['candidates_in_bank']]
        observations = read(folder / 'observations.json')
        references = [v for r in observations for v in r['references'].values()]
        sources = Counter(v.get('source', 'missing') for r in observations
                          for v in r['provenance'].values())
        report.update({k: v for k, v in native.items() if k != 'results'})
        report['native_bank'] = dict(
            probe_documents=len(bindings), candidate_occurrences=len(present),
            candidate_occurrences_present=sum(present),
            candidate_coverage=sum(present) / max(1, len(present)),
            all_candidates_present=sum(all(r['native_bank']['candidates_in_bank'])
                                       for r in bindings),
            empty_banks=sum(not r['native_bank']['ids'] for r in bindings))
        report['mention_audit'] = dict(
            count=len(references), valid=sum(map(valid_reference, references)),
            missing=sum(v is None for v in references),
            mint_sentinel=sum(v == -1 for v in references),
            zero_sentinel=sum(v == 0 for v in references), sources=dict(sources))
        report['checkpoint_sha256'] = (digest(folder / 'final.ckpt')
                                       if (folder / 'final.ckpt').exists() else None)
        if (folder / 'failure.json').exists():
            report['post_result_failure'] = read(folder / 'failure.json')
        reports[name] = report
    result = dict(source_check=check, warmup_checkpoint_sha256=warm, branches=reports,
                  on_off_checkpoints_equal={
                      corpus: bool(reports[f'identity-{corpus}-on'].get('checkpoint_sha256'))
                      and reports[f'identity-{corpus}-on'].get('checkpoint_sha256')
                      == reports[f'identity-{corpus}-off'].get('checkpoint_sha256')
                      for corpus in ('balanced', 'biased')},
                  rule_retired=False,
                  note='Unresolved choices remain incorrect in unconditional accuracy. '
                       'Missing candidates or unresolved choices do not establish a '
                       'successful chance-level control. No rule is retired by this report.')
    target = Path(__file__).resolve().parent / 'validation/identity-summary.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(target)


if __name__ == '__main__':
    main()
