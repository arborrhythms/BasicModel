import json,inspect,os
import torch,IdentityRows
original=IdentityRows.coordinates
def observed(owner,point,**kwargs):
 try:return original(owner,point,**kwargs)
 except ValueError:
  derived=getattr(owner.similarity_codebook,'mereology',None)
  d=dict(minimum=float(point.min()),maximum=float(point.max()),bad_indices=(((point<0)|(point>1)).nonzero().tolist()),width=point.shape[-1],percept_width=getattr(derived,'percept_width',None),band_end=getattr(derived,'percept_event_width',None),frames=[])
  for f in inspect.stack()[:12]:
   row=dict(file=os.path.basename(f.filename),line=f.lineno,function=f.function)
   for k in ('node','request','clause'):
    v=f.frame.f_locals.get(k)
    if v is not None:row[k]=repr(v)[:4000]
   if f.function == '_sentence_observation':
    model=f.frame.f_locals['self']; b=f.frame.f_locals['b'];entry=f.frame.f_locals['entries'][b]
    row['actions']=entry.actions.tolist()
    row['leaf_ranges']=[(float(x.min()),float(x.max())) for x in entry.leaves]
    row['activations']=entry.activations.tolist()
    row['journal']=[[(float(x.min()),float(x.max())) for x in step] for step in entry.operation_values]
    row['binary']=[r.method_name for r in model.languageSpace._compose_binary_rules]
    row['unary']=[r.method_name for r in model.languageSpace._compose_unary_rules]
   d['frames'].append(row)
  with open(os.environ['CUBE_TRACE'],'a') as output:output.write(json.dumps(d)+'\n')
  raise
IdentityRows.coordinates=observed
