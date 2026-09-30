"""Join the named, already measured XOR receipts without dropping any attempt."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = [('baseline', 'baseline-head', 'baseline-candidate'),
          ('Z', 'z-head', 'z-candidate'),
          ('AE', 'ae-head', 'ae-candidate'),
          ('AF', 'af-head', 'af-candidate'),
          ('AH', 'ah-head', 'ah-candidate'),
          ('AI', 'ai-head', 'ai-candidate'),
          ('final', 'final-xor-head', 'final3-xor-candidate')]


def summarize(label):
    summary = json.loads((HERE / label / 'summary.json').read_text())
    result = []
    exact_trial = 0
    for group in summary['groups']:
        exact = group['selector'].endswith('::test_mm20m_xor_exact_roundtrip')
        if exact:
            exact_trial += 1
        for node in group['selected']:
            outcome = group['outcomes'].get(node, group['reason'])
            diagnostics = [dict(exit_code=d['returncode'], peak_gib=d['peak_memory_bytes'] / 2**30)
                           for d in group['diagnostic_only']]
            result.append(dict(nodeid=node, exact_trial=exact_trial if exact else None,
                               outcome=outcome, peak_gib=group['peak_memory_bytes'] / 2**30,
                               diagnostics=diagnostics))
    assert exact_trial == 15
    assert len(result) == 49
    return summary, result


def display(row):
    value = f"{row['outcome']}; {row['peak_gib']:.2f} GiB"
    for diagnostic in row['diagnostics']:
        value += f"; diagnostic exit {diagnostic['exit_code']} ({diagnostic['peak_gib']:.2f} GiB)"
    return value


def main():
    tables = {}
    for stage, head, candidate in STAGES:
        if not all((HERE / label / 'summary.json').exists() for label in (head, candidate)):
            continue
        hs, hr = summarize(head)
        cs, cr = summarize(candidate)
        assert [(r['nodeid'], r['exact_trial']) for r in hr] == [(r['nodeid'], r['exact_trial']) for r in cr]
        tables[stage] = dict(HEAD=head, candidate=candidate, head_outcomes=hs['named_pytest_outcomes'],
                             candidate_outcomes=cs['named_pytest_outcomes'],
                             head_exact=hs['guarded_roundtrip_outcomes'],
                             candidate_exact=cs['guarded_roundtrip_outcomes'])
        lines = [f'# XOR: {stage}, every named proof', '',
                 'All guarded attempts are reported. A guard stop remains a guard stop even when its separate unguarded diagnostic passes. Fifteen exact round trips per tree are included, with no seed selected.', '',
                 f'[HEAD values and raw links]({head}/table.md); [candidate values and raw links]({candidate}/table.md).', '',
                 '| Proof | HEAD | Candidate |', '|---|---|---|']
        for h, c in zip(hr, cr):
            name = h['nodeid'].removeprefix('test/')
            if h['exact_trial'] is not None:
                name += f" — trial {h['exact_trial']}/15"
            lines.append(f'| {name} | {display(h)} | {display(c)} |')
        lines += ['', 'XOR_grammar and the intermittent MM_20M_xor exact-round-trip cause remain item 6.9; neither is repaired in this pass.', '']
        (HERE / f'xor-{stage.lower()}-comparison.md').write_text('\n'.join(lines))
    (HERE / 'xor-comparison-index.json').write_text(json.dumps(tables, indent=2) + '\n')
    print('Completed stage tables:', ', '.join(tables))


if __name__ == '__main__':
    main()
