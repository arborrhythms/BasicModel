import json
import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path.cwd()/'bin'))
from Models import BasicModel
out=Path('doc/benchmarks/2026-10-01-stage1-and-suite-trim/stage1/repairs/native-journal')
active=torch.tensor([[True]*6+[False]*250, [True]*5+[False]*251])
ids=torch.tensor([[0,0,1,1,1,1]+[-1]*250, [0,0,0,1,1]+[-1]*251])
try:
    columns,width=BasicModel._sentence_journal_layout(active,ids,16,4864)
    assert width==28
    assert columns[0,:18].tolist()==[0,1,2,3,4,5,0,1,2,3,4,5,6,7,8,9,10,11]
    assert columns[1,:15].tolist()==[0,1,2,3,4,5,6,7,8,0,1,2,3,4,5]
    assert columns[:,768:784].tolist()==[list(range(12,28))]*2
    assert columns[:,800:816].tolist()==[list(range(12,28))]*2
    result={'passed':True,'journal_slots':width,'full_trace_slots':4864}
except BaseException as e:
    result={'passed':False,'type':type(e).__name__,'message':str(e)}
    (out/'compact-layout-red.json').write_text(json.dumps(result,indent=2)+'\n')
    raise
(out/'compact-layout-green.json').write_text(json.dumps(result,indent=2)+'\n')
