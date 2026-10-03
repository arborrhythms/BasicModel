"""The displayed class count must use the four saved answers."""
import json
from pathlib import Path

here=Path(__file__).resolve().parent
probe=json.loads((here/'probe.json').read_text())
table=(here.parent/'repaired-xor/candidate/table.md').read_text()
row=next(line for line in table.splitlines() if 'TestXorGrammarLearnsXor' in line)
print('expected:',probe['expected_summary'])
print('actual:',row)
assert probe['expected_summary'] in row
