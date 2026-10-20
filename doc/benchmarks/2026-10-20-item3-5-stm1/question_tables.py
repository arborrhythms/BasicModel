"""Render one native question table per generated or decomposition question."""
import argparse
import json
from pathlib import Path

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--generated')
parser.add_argument('--decomposition')
args = parser.parse_args()

def cell(value):
    return '`' + json.dumps(value, ensure_ascii=False).replace('|', '\\|') + '`'

if args.generated:
    name = args.generated
    lines = [f'# G1–G3 in {name}', '',
        'Unchanged generated assertions on the native Part scorer; forced mechanism cases, '
        'not learned reference accuracy. One table follows for each request.', '']
    for path in sorted((root/('generated-'+name)).glob('*/*.json')):
        row = json.loads(path.read_text())
        lines += [f'## {path.parent.name}, case {path.stem}', '', '| Measurement | Result |', '|---|---|']
        lines += [f'| {key.replace("_", " ")} | {cell(value)} |' for key, value in row.items() if key != 'case']
        lines += ['', f'[Raw case]({path.relative_to(root)}).', '']
    (root/('generated-'+name+'-questions.md')).write_text('\n'.join(lines))

if args.decomposition:
    name = args.decomposition
    lines = [f'# §6.10 decomposition, attempt {name}', '',
        'Both original lengths use their original parameter snapshots and seeds '
        '(4199722147 and 2428440506), fixed budgets (512 and 768), forced scopes, '
        'assertions and R+A keep law. Gains are all 0 or all .5; both arms keep '
        'prime on and φ=.5. No extra training or learning claim.', '']
    total = correct = 0
    for arm in ('zero', 'enabled'):
        run = 'bind-decomposition-'+arm+'-stm1-'+name
        for path in sorted((root/(run+'-reports')).rglob('questions.jsonl')):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                total += 1
                correct += bool(row['bound_correct'] and row['chain_correct'] and not row['open'])
                lines += [f"## {arm}, length {row['expected_chain_length']}, document {row['document']}", '',
                    '| Measurement | Result |', '|---|---|']
                for key in ('answer','bound_correct','expected_chain_length','inference_count','chain_correct','work','open','committed_binding','operations'):
                    lines.append(f'| {key.replace("_", " ")} | {cell(row[key])} |')
                lines += ['', f'[Question records]({path.relative_to(root)}); [run]({run}/report.html).', '']
    lines[2:2] = [f'**{correct}/{total} companion questions have the correct binding and proof chain.**', '']
    (root/('decomposition-'+name+'.md')).write_text('\n'.join(lines))
