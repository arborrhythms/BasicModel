"""Capture first native language graph on an explicit source snapshot; no training."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
root=args.root.resolve(); out=args.output.resolve(); out.mkdir(exist_ok=False)
sys.path[:0]=[str(root/'bin'),str(root/'test')]
os.environ.pop('BASIC_SEED',None)
os.environ.update(BASICMODEL_DEVICE='cpu', MODEL_COMPILE='eager',
                  BASIC_AUTOLOAD='false', BASIC_AUTOSAVE='false')
import torch
from torch._dynamo.backends import registry
original=registry.lookup_backend('eager')
serial=0

def capture(gm, examples, **kwargs):
    global serial
    serial+=1
    def tensors(value):
        if isinstance(value,torch.Tensor):
            shape=[d if isinstance(d,int) else str(d) for d in value.shape]
            hints=[d if isinstance(d,int) else d.node.hint for d in value.shape]
            size=math.prod(hints)*value.element_size() if all(isinstance(d,int) for d in hints) else None
            return [dict(shape=shape,hints=hints,dtype=str(value.dtype),bytes=size)]
        if isinstance(value,(list,tuple)):
            return [a for x in value for a in tensors(x)]
        return []
    nodes=[dict(name=n.name,op=n.op,target=str(n.target),
                values=tensors(n.meta.get('example_value',n.meta.get('val'))),
                stack=n.meta.get('stack_trace','')) for n in gm.graph.nodes]
    language=any('stage_cs_lang' in n['stack'] for n in nodes)
    path=out/f'graph-{serial:03}'
    path.with_suffix('.json').write_text(json.dumps(dict(language=language,nodes=nodes),indent=2)+'\n')
    path.with_suffix('.py.txt').write_text(gm.code)
    if language:
        print('Captured first language graph; stopping before execution or training.',flush=True)
        raise SystemExit(73)
    return original(gm,examples,**kwargs)
registry._COMPILER_FNS['eager']=capture
import Models
import util
before=Models.BasicModel._run_sentence_word_bricks

def describe(model, words, active, ids, *rest):
    counts=[{str(int(i)):int(((row==i)&mask).sum()) for i in row[mask].unique()}
            for row,mask in zip(ids,active)]
    (out/'layout.json').write_text(json.dumps(dict(batch=words.shape[0],width=words.shape[1],
        sentence_word_counts=counts,lang_shapes=[list(x.shape) for x in rest[3]],seed=None),indent=2)+'\n')
    return before(model,words,active,ids,*rest)
Models.BasicModel._run_sentence_word_bricks=describe
config=root/'data/BasicModel_answers_tied_benchmark.xml'
util.init_config(path=str(config),defaults_path=str(root/'data/model.xml'))
arch=util.TheXMLConfig.data.get('architecture',{})
Models.ModelFactory._run_hydrated(str(config),arch,arch.get('data',{}),arch.get('training',{}))
raise AssertionError('Expected graph-only stop')
