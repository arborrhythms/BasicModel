"""Apply the prepared closing repair only after the complete baseline receipt."""
import ast
import difflib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'test'))
from bounded_tests import source_snapshot

baseline = json.loads((HERE / 'baseline-candidate' / 'result.json').read_text())
assert baseline['reason'] != 'running' and len(baseline['groups']) == 29
assert len(baseline['diagnostic_only']) == 21
assert source_snapshot(ROOT) == json.loads((HERE / 'incoming-source.json').read_text())
patch = []
for proposal in sorted((HERE / 'z-proposal').glob('*/*.py')):
    relative = proposal.relative_to(HERE / 'z-proposal')
    target = ROOT / relative
    before = target.read_text() if target.exists() else ''
    after = proposal.read_text()
    ast.parse(after)
    patch.extend(difflib.unified_diff(before.splitlines(keepends=True),
                                     after.splitlines(keepends=True),
                                     fromfile='a/' + str(relative), tofile='b/' + str(relative)))
    target.write_text(after)
(HERE / 'z-initial-repair.patch').write_text(''.join(patch))
(HERE / 'z-initial-source.json').write_text(json.dumps(source_snapshot(ROOT), indent=2) + '\n')
print('Applied Z and its recorded phrase-storage port, after the complete frozen baseline.')
