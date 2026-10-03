from pathlib import Path
import json,sys,traceback
ROOT=Path(__file__).resolve().parents[4];sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import recon_bench
out=[]
for name in ('XOR_grammar','MM_grammar','xor'):
 try:
  m,dev,lr,batch=recon_bench._build_model(str(ROOT/'data'/f'{name}.xml'))
  out.append(dict(config=name,batch=batch,scope=getattr(m,'reconstruction_scope',None),reconstruct_in_loop=getattr(m,'reconstruct_in_loop',None)))
  m.End()
 except Exception as e:out.append(dict(config=name,error=repr(e),traceback=traceback.format_exc()))
Path(__file__).with_suffix('.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
