"""Hold historical tables until the final reconstruction campaign releases resources."""
import json,os,sys,time
from pathlib import Path
here=Path(__file__).resolve().parent
label,root,*dependencies=sys.argv[1:]
sys.path.insert(0,str(Path(root)/'test'))
from bounded_tests import source_snapshot
source=source_snapshot(Path(root))
print(label+': waiting for final reconstruction and '+repr(dependencies),flush=True)
while not (here/'final-reconstruction-complete.json').exists():time.sleep(2)
for name in dependencies:
 p=here/name/'result.json'
 while not p.exists() or json.loads(p.read_text()).get('reason')=='running':time.sleep(2)
assert source_snapshot(Path(root))==source
os.execv(sys.executable,[sys.executable,str(here/'run_xor.py'),label,root])
