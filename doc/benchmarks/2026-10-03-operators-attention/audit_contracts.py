"""Observation-only comparison with the original published source archive."""
import ast, hashlib, json, subprocess, zipfile
from pathlib import Path
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
old=zipfile.ZipFile(HERE/'before.zip')
seed_changes=[];xml_seed_changes=[];capacity_changes=[]
def seed_calls(source):
 return sorted(ast.get_source_segment(source,node) for node in ast.walk(ast.parse(source))
  if isinstance(node,ast.Call) and 'seed' in ast.unparse(node.func).lower())
for path in old.namelist():
 p=ROOT/path
 if not p.exists():continue
 before=old.read(path).decode();after=p.read_text()
 if path.endswith('.py') and seed_calls(before)!=seed_calls(after):
  seed_changes.append(dict(path=path,old=seed_calls(before),new=seed_calls(after)))
 if path.startswith('data/') and path.endswith('.xml'):
  try:a,b=ET.fromstring(before),ET.fromstring(after)
  except ET.ParseError:continue
  def values(root,tags):
   return [(section.tag,node.tag,(node.text or '').strip()) for section in root.iter() for node in section if node.tag in tags]
  if values(a,{'seed'})!=values(b,{'seed'}):xml_seed_changes.append(path)
  if values(a,{'nVectors','activeVectors','nInput','nInputDim','nOutput','nOutputDim','nDim','serialWordCapacity','serialResidualPartCapacity','reconstructionBasisLimit'})!=values(b,{'nVectors','activeVectors','nInput','nInputDim','nOutput','nOutputDim','nDim','serialWordCapacity','serialResidualPartCapacity','reconstructionBasisLimit'}):capacity_changes.append(path)
protected=['test/test_explicit_dimensions.py','test/test_mm_xor.py','test/test_reconstruction_roundtrip.py',
 'test/bounded_tests.py','Makefile','pytest.ini','data/eval/nanochat_grammar_gate.json']
unchanged={path:old.read(path)==(ROOT/path).read_bytes() for path in protected if path in old.namelist()}
result=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 worktrees=subprocess.check_output(['git','worktree','list','--porcelain'],cwd=ROOT,text=True),
 changed_python_seed_calls=seed_changes,changed_xml_seeds=xml_seed_changes,changed_declared_capacities=capacity_changes,
 protected_files_unchanged=unchanged,guards=dict(worker_gib=8,worker_timeout_seconds=1800,aggregate_gib=28,workers=10,cpu_headroom='unchanged bounded worker policy'))
(HERE/'contracts-audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
