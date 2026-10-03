"""Move parked-class and standalone experiment tests to their owners."""
import ast,re,textwrap,json
from port_ledger import ROOT,HERE,definitions,record,remove_definitions
moves={
 'bin/Legacy.py': [('test/test_basicmodel.py','TestQKVAttentionLayer'),('test/test_basicmodel.py','TestMemory'),('test/test_conceptual_introspection.py','TestIntrospectionLayers'),('test/test_mereology.py','TestCopyLayer')],
 'bin/etc/SPNN.py':[('test/test_basicmodel.py','TestSPNN')],
 'bin/etc/SigmaPi.py':[('test/test_basicmodel.py','TestSigmaPi'),('test/test_sigmapi.py','TestLogicalFunctionNet')],
 'bin/etc/SymPercept.py':[('test/test_basicmodel.py','TestSymPercept')],
}
for target,items in moves.items():
 p=ROOT/target;s=p.read_text();blocks=[];oldcases=[]
 for filename,cls in items:
  old=remove_definitions(filename,[cls])[cls]
  for name,b in definitions(old).items():
   if '::test_' in name:oldcases.append((filename,name,b))
  new=old.replace('Legacy.','').replace('Models.TheDevice','TheDevice')
  new=re.sub(r'^    @pytest.mark.slow\n','',new,flags=re.M)
  new=re.sub(r'^        from (SPNN|SigmaPi|SymPercept) import .+\n','',new,flags=re.M)
  blocks.append(new)
 if target=='bin/Legacy.py':
  old=remove_definitions('test/test_surface_schema.py',['test_copy_swap_use_elision_template'])['test_copy_swap_use_elision_template']
  new=old.replace('def test_copy_swap_use_elision_template():','def test_copy_swap_use_elision_template(self):').replace('    from Legacy import CopyLayer, SwapLayer\n','')
  blocks.append('class TestParkedSurfaceSchema(unittest.TestCase):\n'+textwrap.indent(new,'    '))
  oldcases.append(('test/test_surface_schema.py','test_copy_swap_use_elision_template',old))
  blocks.append('''class TestParkedLayerSelfTests(unittest.TestCase):
    def test_qkv(self):
        QKVAttentionLayer.test()

    def test_memory(self):
        Mem.test()

    def test_decision_boundary(self):
        DecisionBoundaryLayer.test()
''')
  header='\n\nif __name__ == "__main__":\n    import unittest\n    import warnings\n    import matplotlib\n    matplotlib.use("Agg")\n    from Language import GRAMMAR_LAYER_CLASSES\n    util.init_device("cpu")\n\n'
 else:
  # Direct execution must find the live bin modules without pytest's sys.path.
  doc=ast.parse(s).body[0];lines=s.splitlines(keepends=True)
  bootstrap='''
import sys
from pathlib import Path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    if "--demo" not in sys.argv:
        import os
        os.environ.setdefault("BASICMODEL_DEVICE", "cpu")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("MPLBACKEND", "Agg")
'''
  lines[doc.end_lineno:doc.end_lineno]=[bootstrap];s=''.join(lines)
  s=re.sub(r"if __name__ == (['\"])__main__\1:", 'if __name__ == "__main__" and "--demo" in sys.argv:',s,count=0)
  # Restore the unconditional bootstrap; only the former experiment is a demo.
  s=s.replace('if __name__ == "__main__" and "--demo" in sys.argv:\n    sys.path.insert', 'if __name__ == "__main__":\n    sys.path.insert',1)
  header='\n\nif __name__ == "__main__" and "--demo" not in sys.argv:\n    import unittest\n    from util import TheDevice\n    util.init_device("cpu")\n\n'
  s=s.replace('Run directly to execute the experiment:  python SigmaPi.py','Run ``python bin/etc/SigmaPi.py`` for inline tests; add ``--demo`` for the experiment.')
 s+=header+textwrap.indent('\n\n'.join(blocks),'    ')+'\n    unittest.main()\n'
 ast.parse(s);p.write_text(s)
 for filename,name,old in oldcases:
  destination='TestParkedSurfaceSchema::'+name if name=='test_copy_swap_use_elision_template' else name
  record(filename,name,old,[(target,destination)],'Move parked implementation tests beside their owner; assertions unchanged; standalone unittest entry point, outside pytest.', ['October 1 hand-off part 4 items 2–3', 'pre-trim-source.zip'])
# Move the three self-test calls and preserve the complete original/new owner bodies.
p=ROOT/'bin/Layers.py';s=p.read_text();old=definitions(s)['test']
for name in ('QKVAttentionLayer','Mem','DecisionBoundaryLayer'):
 s=s.replace(f'    {name}.test()\n','',1)
p.write_text(s)
record('bin/Layers.py','test',old,[('bin/Layers.py','test'),('bin/Legacy.py','TestParkedLayerSelfTests')], 'Move the three parked-class self-test calls to Legacy.py; live-layer calls remain.', ['October 1 hand-off part 4 item 2'])
# Remove imports used solely by the moved tests; keep live mereology helpers.
p=ROOT/'test/test_basicmodel.py';s=p.read_text().replace('import Legacy\n','');s=s.replace('  - SPNN.py    classical neural network\n  - SigmaPi.py product-sum network\n  - SymPercept.py  bidirectional linear learning\n','')
for text in ('SPNN.py -- Classical neural network','SigmaPi.py -- Product-sum network','SymPercept.py -- Bidirectional linear learning'):
 s=s.replace('# ---------------------------------------------------------------------------\n# '+text+'\n# ---------------------------------------------------------------------------\n','')
p.write_text(s)
p=ROOT/'test/test_conceptual_introspection.py';s=p.read_text();s=re.sub(r'from Legacy import \(.*?\)\n','',s,flags=re.S);s=s.replace('    GRAMMAR_LAYER_CLASSES,\n','');p.write_text(s)
p=ROOT/'test/test_mereology.py';s=p.read_text().replace('from Legacy import CopyLayer, SwapLayer  # noqa: E402  (parked 2026-07-17)\n','').replace('    GRAMMAR_LAYER_CLASSES,\n','').replace('Also re-asserts retained Phase 1b utilities (`CopyLayer`,\n`_gaussian_kernel_overlap`, `ste_answer`) that survived the revert.','Also exercises `_gaussian_kernel_overlap` and `ste_answer`; parked grammar\nclass tests live in Legacy.py.');p.write_text(s)
print('Moved 16 Legacy cases, 7 experiment cases, and the three Layers self-test calls; parked implementations retained.')
