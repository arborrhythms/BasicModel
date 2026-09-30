"""Preserve the review's configuration and seed constraints in a receipt."""
import ast,hashlib,json,subprocess
from pathlib import Path
h=Path(__file__).resolve().parent;r=h.parents[2]
base=json.loads((h/'baseline-candidate/source-manifest.json').read_text())['validated_source']
# The bounded runner reports both its aggregate digest and individual files.
configuration_changes=[name for name, digest in base.items() if name.startswith('data/') and name.endswith(('.xml','.xsd','.grammar')) and hashlib.sha256((r/name).read_bytes()).hexdigest()!=digest]
violations=[]
for p in sorted((r/'test').glob('test_item7_*.py')):
 for n in ast.walk(ast.parse(p.read_text())):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('seed','manual_seed','manual_seed_all'):
   violations.append(dict(file=str(p.relative_to(r)),line=n.lineno,kind='seed'))
head_configs={}
for name in ('MM_xor.xml','MM_grammar.xml','XOR_grammar.xml'):
 path='data/'+name
 head=subprocess.check_output(['git','show','HEAD:'+path],cwd=r)
 current=(r/path).read_bytes()
 expected=head.replace(b'    <propertyBasis>true</propertyBasis>\n',b'')
 head_configs[path]=dict(head_sha256=hashlib.sha256(head).hexdigest(),
     current_sha256=hashlib.sha256(current).hexdigest(),
     byte_identical_except_retired_propertyBasis=current==expected)
 assert current==expected,path
result=dict(item7_seed_calls=violations, configuration_changes_since_round3_baseline=configuration_changes,
            required_head_config_comparison=head_configs)
(h/'constraints-audit.json').write_text(json.dumps(result,indent=2)+'\n')
assert not violations and not configuration_changes
