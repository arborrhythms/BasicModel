"""Read-only replay of the native stage3 restore, with timed stack captures."""
from pathlib import Path
from datetime import datetime,timezone
import faulthandler,json,os,sys,time,traceback
ROOT=Path.cwd();OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch,pytest
from identity_experiment import make_model,restore_warmup
from Models import BaseModel
restore=BaseModel.load_weights
torch.set_num_threads(1)
started=time.monotonic()
record={'source':os.popen('git rev-parse HEAD').read().strip(),'started':datetime.now(timezone.utc).isoformat(),'checkpoint':str(OUT.parent/'stage3/order.ckpt'),'training_updates':0}
(OUT/'start.json').write_text(json.dumps(record,indent=2)+'\n')
with (OUT/'stacks.txt').open('w') as stacks,pytest.MonkeyPatch.context() as patch:
 model=make_model(OUT,patch)
 faulthandler.dump_traceback_later(30,repeat=True,file=stacks)
 try:
  drift=restore_warmup(model,OUT.parent/'stage3/order.ckpt',restore)
  record.update(completed=True,drift=drift)
 except BaseException:
  record.update(completed=False,error=traceback.format_exc());raise
 finally:
  faulthandler.cancel_dump_traceback_later();record['elapsed_seconds']=time.monotonic()-started
  (OUT/'run.json').write_text(json.dumps(record,indent=2)+'\n');model.End()
