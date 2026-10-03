import sys, importlib.util
from pathlib import Path
root=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(root/'bin'),str(root/'test')]
sys.argv=['objective_conflicts_probe.py','--config','XOR_grammar','--output',str(Path(__file__).parent/'repairs/observer-contract-before'),'--validate-only']
spec=importlib.util.spec_from_file_location('observer',root/'test/objective_conflicts_probe.py')
observer=importlib.util.module_from_spec(spec)
try:spec.loader.exec_module(observer)
except SystemExit:pass
from test_mm_xor import _fresh_model
model,_,data=_fresh_model(str(root/'data/XOR_grammar.xml'))
observer.P.groups(model)
