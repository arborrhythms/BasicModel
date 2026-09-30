"""Fixture integration after the ordered item-7 design changes; not the final sweep."""
import importlib.util
import os
import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rename_runner',HERE/'run_rename_sweep.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.HERE=HERE/'affected'
runner.HERE.mkdir(exist_ok=False)
changed=subprocess.check_output(['git','diff','--name-only','--','test'],cwd=ROOT,text=True).splitlines()
new=subprocess.check_output(['git','ls-files','--others','--exclude-standard','test'],cwd=ROOT,text=True).splitlines()
extra=['test/test_subspace_what_stm_contract.py','test/test_idea_decode_d3.py',
       'test/test_fineweb_preflight.py','test/test_priming_energy.py',
       'test/test_runtime_split_ingestion.py','test/test_config_matrix.py']
selected=sorted({p for p in changed+new+extra if Path(p).name.startswith('test_') and p.endswith('.py') and (ROOT/p).exists()})
(runner.HERE/'selectors.txt').write_text('\n'.join(selected)+'\n')
original=runner.bounded.run_suite
def run_suite(**kwargs):
    kwargs['selectors']=selected
    return original(**kwargs)
runner.bounded.run_suite=run_suite
raise SystemExit(runner.main())
