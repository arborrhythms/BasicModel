"""Save a bounded diagnostic's command, output, exit status and source hashes."""
import json
import os
from pathlib import Path
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'test'))
import bounded_tests as bounded

folder = HERE/'probes'/sys.argv[1]
folder.mkdir(parents=True, exist_ok=False)
bounded.write_json(folder/'source.json', bounded.source_snapshot(ROOT))
with zipfile.ZipFile(folder/'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in bounded.source_snapshot(ROOT):
        archive.write(ROOT/name, name)
environment = os.environ.copy()
environment.update(MODEL_COMPILE=os.environ.get('PROBE_COMPILE','none'), BASICMODEL_DEVICE='cpu',
                   KMP_DUPLICATE_LIB_OK='TRUE', BASIC_AUTOLOAD='0')
result = bounded.run_guarded([sys.executable, '-m', 'pytest', '-q', *sys.argv[2:]],
    cwd=ROOT, env=environment, log_path=folder/'run.log',
    memory_bytes=8*bounded.GIB, timeout=1800)
bounded.write_json(folder/'process.json', result)
print(json.dumps(result))
print((folder/'run.log').read_text()[-18000:])
