"""Clamp inactive pipeline fill/drain addresses before journal gathers."""
import json,difflib
from port_ledger import HERE, ROOT, definitions
p=ROOT/'bin/Models.py';source=p.read_text()
old=definitions(source)['BasicModel::_run_tensor_peer_word_pipeline']
a='''            journal_columns = torch.cat((
                (3 * index + torch.arange(3, device=words.device))[None].expand(B, -1),'''
b='''            # Pipeline fill/drain ticks carry an out-of-range word index.
            # They have no active grammar row, but gather/scatter addresses
            # must still be legal before the row mask makes the write a no-op.
            journal_word = index.clamp(0, width - 1)
            journal_columns = torch.cat((
                (3 * journal_word + torch.arange(3, device=words.device))[None].expand(B, -1),'''
assert a in old
new=old.replace(a,b,1)
(HERE/'journal-drain-repair.json').write_text(json.dumps(dict(
    failing_probes=['mps-closing-check/worker-000.log','mps-closing-check/worker-001.log'],
    reason='The compact per-word journal introduced a gather before the existing inactive-row mask. The two initial pipeline delay ticks use word indices -2 and -1; only their physical address is bounded, not their activity, recorded choice or numerical value.',
    old=old,new=new),indent=2)+'\n')
(HERE/'journal-drain-repair.patch').write_text(''.join(difflib.unified_diff(
 old.splitlines(True),new.splitlines(True),fromfile='old/bin/Models.py',tofile='new/bin/Models.py')))
p.write_text(source.replace(old,new,1))
