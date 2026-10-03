import os,sys,json
from pathlib import Path
root=Path(__file__).resolve().parents[3];sys.path.insert(0,str(root/'bin'))
os.environ['MODEL_COMPILE']='none'
from util import init_config,init_device,TheXMLConfig
import util,Models,Language
from data import TheData
from WhereRegistry import WhereRegistry
init_device('cpu');util.TheCompileBackend='none'
for name in sys.argv[1:] or ['MM_math']:
 path=str(root/'data'/f'{name}.xml');init_config(path=path,defaults_path=str(root/'data/model.xml'))
 dat=dict(TheXMLConfig.get('architecture.data'));TheData.load(dat['dataset'],dat=dat)
 model,_=Models.BaseModel.from_config(path,data=TheData)
 print('CAPACITY',json.dumps(dict(name=name,raw=model.inputSpace.inputShape,output=model.inputSpace.outputShape,byte_length=TheData.inputLength,serial=model.serial,input_range=model.where_registry.slices['input'])),flush=True)
 original=WhereRegistry.intervals
 def observed(registry,name,indices):
  if name=='input':print('OCCURRENCE',json.dumps(dict(maximum=int(indices.max()),minimum=int(indices.min()),range=registry.slices[name])),flush=True)
  return original(registry,name,indices)
 WhereRegistry.intervals=observed
 try:
  items,target=next(iter(TheData.data_loader(split='train',num_streams=1)))
  for presentation in range(3):
   print("PRESENTATION",presentation,flush=True)
   model._lex_embed_stem(model.inputSpace.prepInput(items))
 finally:
  WhereRegistry.intervals=original;model.End();model.symbolSpace.soft_reset()
