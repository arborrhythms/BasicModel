"""Continue predeclared first draws after inspecting a retained failure."""
import json
from pathlib import Path
import sys

receipt = Path(__file__).resolve().parents[1]
arm = sys.argv[1]
prior = receipt / f'fresh-final-{arm}.json'
cases = json.loads(prior.read_text())
assert cases and cases[-1]['exit'] != 0
start = len(cases) + 1
assert start <= 6
path = receipt / 'fresh_arrivals.py'
source = path.read_text().replace('cases = []',
    f'cases = json.loads(Path({str(prior)!r}).read_text())').replace(
    'range(1, 7)', f'range({start}, 7)')
exec(compile(source, str(path), 'exec'), {'__file__': str(path), '__name__': '__main__'})
