"""Save complete old and new files and bodies for every test port in this round."""
import ast,hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent;ROOT=R.parents[3]

def bodies(source):
 lines=source.splitlines(keepends=True);result={}
 def walk(nodes,prefix=''):
  for n in nodes:
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    name=prefix+n.name
    first=min([n.lineno]+[x.lineno for x in n.decorator_list])-1
    result[name]=''.join(lines[first:n.end_lineno])
    if isinstance(n,ast.ClassDef):walk(n.body,name+'.')
 walk(ast.parse(source).body)
 return result

index=[];changed=[]
with zipfile.ZipFile(R/'before.zip') as before,zipfile.ZipFile(R/'test-ports-old-new.zip','w',zipfile.ZIP_DEFLATED) as archive:
 paths=set(n for n in before.namelist() if n.startswith('test/') and n.endswith('.py'))
 paths|={str(p.relative_to(ROOT)) for p in (ROOT/'test').rglob('*.py')}
 for path in sorted(paths):
  old=before.read(path).decode() if path in before.namelist() else ''
  new=(ROOT/path).read_text() if (ROOT/path).exists() else ''
  if old==new:continue
  changed.append(path)
  archive.writestr('old/'+path,old);archive.writestr('new/'+path,new)
  a,b=bodies(old),bodies(new)
  for name in sorted(a.keys()|b.keys()):
   if a.get(name)==b.get(name):continue
   kind='removed' if name not in b else 'added' if name not in a else 'ported'
   index.append(dict(file=path,name=name,disposition=kind,old=a.get(name),new=b.get(name)))
(R/'test-ports-old-new.json').write_text(json.dumps(dict(files=changed,bodies=index),indent=2)+'\n')
(R/'test-port-index.md').write_text('# Complete test ports\n\n`test-ports-old-new.json` retains complete old/new bodies (including fixture helpers and decorators). `test-ports-old-new.zip` retains complete old/new files. Retired path reasons are in `mode-test-dispositions.json`, `remaining-test-deletions.json`, and `extra-port-deletions.json`. Renames have one removed and one added body.\n\n| File | Definition | Disposition |\n|---|---|---|\n'+''.join(f"| `{v['file']}` | `{v['name']}` | {v['disposition']} |\n" for v in index))
print(len(changed),'changed test files;',len(index),'complete body pairs')
