"""Install only the mixing reconstruction-bank boundary for stage 1."""
import json,difflib,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'suite-trim'));from port_ledger import definitions
p=ROOT/'bin/Models.py';s=p.read_text();old=definitions(s)['BasicModel::_lex_embed_stem']
a='''        self._stage_snapshot_bytes()
        if getattr(self, "reconstruct_in_loop", False):'''
b='''        self._stage_snapshot_bytes()
        self._stage_mixing_reconstruction_bank()
        if getattr(self, "reconstruct_in_loop", False):'''
assert a in old
new=old.replace(a,b,1);helper=(HERE/'mixing-bank.draft.py').read_text()
(HERE/'mixing-bank-repair.json').write_text(json.dumps(dict(
 failing_probes=['mixing-bank-before/worker-000.log','../stage1/XOR_grammar-step5a/run.log','../stage1/XOR_grammar-cut/run.log'],
 changes=[dict(name='BasicModel._lex_embed_stem',old=old,new=new),dict(name='BasicModel._stage_mixing_reconstruction_bank',old=None,new=helper)],
 native_word_identity_fields_written=False),indent=2)+'\n')
s=s.replace(old,new,1)
assert '    def _stage_mixing_reconstruction_bank(' not in s
s=s.replace('    def _validate_reconstruction_bank(self):',helper+'    def _validate_reconstruction_bank(self):',1)
p.write_text(s)
