"""Match the existing byte target/bank shape contract without changing bounds."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'suite-trim'));from port_ledger import definitions
p=ROOT/'bin/Models.py';s=p.read_text();old=definitions(s)['BasicModel::_stage_mixing_reconstruction_bank']
new=old.replace('        width = 1\n','        width = 3\n',1).replace('        def tensor_bytes(values):','        def tensor_bytes(values, lookahead=0):',1).replace('data = torch.zeros(B, W, width + 1,','data = torch.zeros(B, W, width + lookahead,',1).replace('= tensor_bytes(surfaces)','= tensor_bytes(surfaces, lookahead=1)',1)
assert old!=new
(HERE/'mixing-bank-target-width-repair.json').write_text(json.dumps(dict(failing_probes=['XOR_grammar-step5a/run.log','XOR_grammar-cut/run.log'],old=old,new=new,reason='Only the candidate bank owns a lookahead byte. The target owns its actual word bytes (minimum padded extent 3); _byte_word_cost supplies the target terminator. Retain the existing dynamic extent bounds and actual scoring masks.'),indent=2)+'\n')
p.write_text(s.replace(old,new,1))
