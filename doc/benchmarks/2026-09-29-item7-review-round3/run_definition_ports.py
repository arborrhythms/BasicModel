import subprocess, sys
from pathlib import Path
here=Path(__file__).resolve().parent
root=here.parents[2]
files=subprocess.check_output(['rg','-l','interpret_word|lookup_word|interpret.forward|ReferenceTable|meta_members|bind_meta|word_obj_meta|interpretations','test','--glob','*.py'],cwd=root,text=True).splitlines()
(here/'port-files.json').write_text(__import__('json').dumps(files,indent=2)+'\n')
backup=here/'pre-port-source';backup.mkdir(exist_ok=True)
for file in files:
    path=backup/file;path.parent.mkdir(exist_ok=True,parents=True)
    if not path.exists():path.write_bytes((root/file).read_bytes())
raise SystemExit(subprocess.call([sys.executable,str(here/'run_checks.py'),'stage-b-port-probes',str(root),*files]))
