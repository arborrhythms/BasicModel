"""Final mechanical test-helper and whitespace ports after config timings."""
from pathlib import Path
import ast,json,re,subprocess,sys
R=Path(__file__).resolve().parent;ROOT=R.parents[3]
assert (R/'mode-final-batches/complete.json').exists()
p=ROOT/'test/test_input_word_cursor.py';s=p.read_text()
assert s.count('    from space_equiv import _p\n')==2
s=s.replace('    from space_equiv import _p\n','    from pathlib import Path\n    _p = Path(__file__).resolve().parents[1]\n')
s=s.replace('        "from space_equiv import _p\\n"','        "sys.path.insert(0, \'bin\')\\n"\n        "from pathlib import Path\\n"\n        "_p = Path.cwd()\\n"')
assert 'from space_equiv import' not in s
p.write_text(s)
# Only the whitespace diagnostics saved before this repair are changed.
changed=[]
for path in sorted(set(re.findall(r'^([^:\n]+):\d+:', (R/'diff-check-before.log').read_text(),re.M))):
 p=ROOT/path;old=p.read_text();new=old.rstrip()+'\n'
 if path=='bin/Legacy.py':new=new.replace('# the wrapper layers are revivable from \n','# the wrapper layers are revivable from\n')
 assert ast.dump(ast.parse(old))==ast.dump(ast.parse(new)),path
 p.write_text(new);changed.append(path)
p=ROOT/'test/test_grammar_separator.py';s=p.read_text();t=s.replace("== b' ' \n","== b' '\n");assert ast.dump(ast.parse(s))==ast.dump(ast.parse(t));p.write_text(t)
(R/'final-mechanical-ports.json').write_text(json.dumps(dict(helper='test/test_input_word_cursor.py: replace the retired helper module’s repository-path constant with pathlib, including its subprocess bin path. The predicates and every assertion stay unchanged.',whitespace_only_ast_identical=changed),indent=2)+'\n')
subprocess.run([sys.executable,str(R/'ports_archive.py')],cwd=ROOT,check=True)
