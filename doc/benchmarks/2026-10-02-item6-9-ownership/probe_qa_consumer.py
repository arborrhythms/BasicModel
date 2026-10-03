import os,sys,tempfile
from pathlib import Path
root=Path(__file__).resolve().parents[3];sys.path[:0]=[str(root/'bin'),str(root/'test')]
os.environ.update(BASICMODEL_DEVICE='cpu',MODEL_COMPILE='none',BASIC_AUTOLOAD='false',BASIC_AUTOSAVE='false')
import torch,Models,util
from data import TheData
from reading_fixtures import use_eager_reading
from _pytest.monkeypatch import MonkeyPatch
patch=MonkeyPatch();use_eager_reading(patch)
with tempfile.TemporaryDirectory() as folder:
 path=Path(folder)/'qa.xml';s=(root/'data/MM_qa.xml').read_text().replace('1024','64').replace('65536','512').replace('8192','512');path.write_text(s)
 util.init_config(str(path),defaults_path=str(root/'data/model.xml'));TheData.load('xor');m,_=Models.BaseModel.from_config(str(path),data=TheData)
 m.global_attention.consume_gate.data.fill_(.3)
 raw=m.inputSpace.prepInput(['hello world']);out=m.forward(raw)[2];out.square().sum().backward()
 print('serial',m.serial,'attention observation',getattr(m,'_global_attention_obs',None) is not None,'gate gradient',m.global_attention.consume_gate.grad,flush=True)
 assert m.global_attention.consume_gate.grad is not None and m.global_attention.consume_gate.grad.abs().sum()>0
 assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.global_attention.scorer.parameters())
