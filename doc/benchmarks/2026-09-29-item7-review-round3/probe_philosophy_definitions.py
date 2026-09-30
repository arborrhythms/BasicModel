"""Check that the current philosophical contract follows the DEF amendment."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
path = HERE.parents[2] / 'doc/Philosophy.md'
text = path.read_text()
stale = [value for value in (
    'META node as a generalisation over both the word-concept',
    'Decision: META concepts',
    "binding table's one-row-per-word law is replaced by the\n   n-ary META",
) if value in text]
result = dict(path='doc/Philosophy.md', sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
              stale_contracts=stale, passed=not stale)
(HERE / sys.argv[1]).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
raise SystemExit(0 if result['passed'] else 1)
