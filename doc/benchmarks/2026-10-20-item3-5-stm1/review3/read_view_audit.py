"""Compare raw paths with the working reads actually recorded at failure."""
import json
from pathlib import Path
import sys


directory = Path(sys.argv[1])
store = json.loads((directory / 'failure-store.json').read_text())
batch = json.loads((directory / 'failure-batch.json').read_text())
parts = {int(key): value['refs'] for key, value in store['parts'].items()}
rows = {row['address'][2]: row for row in store['rows']}
# The captured fixture's supplied answer. No model execution or score changes.
known = set(store['words']['three'])
known |= {obj for obj, word in parts.values() if word in known}
working = {}
for read in map(json.loads, (directory / 'reads.jsonl').read_text().splitlines()):
    if not read['exclude']:
        working[read['index']] = read['after']


def path_to_answer(root, resolved):
    pending, seen = [(root, [])], set()
    while pending:
        reference, path = pending.pop()
        if reference in known:
            return path + [reference]
        if reference in seen or reference in (-1, 0):
            continue
        seen.add(reference)
        if reference in parts:
            pending.extend((ref, path + [reference]) for ref in parts[reference])
        if reference in rows:
            row = rows[reference]
            value = working.get(row['index'], row['raw']) if resolved else row['raw']
            pending.extend((ref[-1], path + [reference]) for ref in value['refs'] if ref)
    return None


report = dict(capture=str(directory), known=sorted(known), questions=[dict(
    source=item['source'], occurrence=item['address'],
    raw_path=path_to_answer(item['row_id'], False),
    working_path=path_to_answer(item['row_id'], True))
    for item in batch['observations'][-8:]])
print(json.dumps(report, indent=2))
