"""Measure Python container sizes at the host readers, never tensor contents.

The source targets come from the detailed inventory. The replay also records
batch state and requires exact equality with the unobserved native lesson.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import math
import torch
from sentence_host_inventory import StateObserver,compare_states
from attention_thinking_certificate import lesson,restore


class Sizes:
    def __init__(self,targets):
        self.targets=set(targets)
        self.values={};self.calls=Counter();self.expected=[];self.codes={};self.names={n for f,n in targets}
    def size(self,key,value):
        if torch.is_tensor(value):
            info=dict(kind='tensor',shape=list(value.shape),size=math.prod(value.shape))
        elif isinstance(value,(dict,list,tuple,set,frozenset)):
            info=dict(kind=type(value).__name__,size=len(value))
        elif isinstance(value,(int,float)) and not isinstance(value,bool):
            info=dict(kind=type(value).__name__,size=value)
        else:return
        old=self.values.get(key)
        if old is None or info['size']>old['size']:self.values[key]=info
    def __call__(self,frame,event,arg):
        code=frame.f_code
        if code not in self.codes:
            pair=(Path(code.co_filename).name,code.co_qualname) if code.co_qualname in self.names else None
            self.codes[code]=pair if pair in self.targets else None
        pair=self.codes[code]
        if pair is None or event!='return':return
        self.calls[pair]+=1
        for name,value in frame.f_locals.items():
            self.size((*pair,name),value)
            if name in ('self','meaning','reading','program','step','context','store','work'):
                fields=vars(value) if hasattr(value,'__dict__') else {}
                for key,child in fields.items():
                    if not key.startswith('_') and not torch.is_tensor(child):self.size((*pair,name+'.'+key),child)
        if event=='return':self.size((*pair,'return'),arg)
        if event=='return' and pair==('Queries.py','ThoughtLTMCapability.implication_rules'):
            v=frame.f_locals
            expected=object.__getattribute__(v['self'],'_ThoughtLTMCapability__expected')
            self.expected.append(dict(length=len(expected),prefix=list(expected[:v.get('index_limit',0)]),
                limit=v.get('index_limit'),rows=v.get('rows'),result_count=len(arg or ())))
    def report(self):
        return dict(calls=[dict(file=f,function=n,count=c) for (f,n),c in self.calls.items()],
                    values=[dict(file=f,function=n,object=k,**v) for (f,n,k),v in self.values.items()],expected=self.expected)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('replay','inventory','compare','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    d=json.loads(args.inventory.read_text());counts=Counter()
    for r in d['host_sites']:counts[r['file'],r['function']]+=r['count']
    sizes=Sizes([p for p,c in counts.most_common(30)]+[('ConceptSpread.py','rule_order'),('Queries.py','ThoughtLTMCapability.implication_rules')])
    observer=StateObserver(args.output)
    torch.set_num_threads(1)
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        restore(torch.load(args.replay/'initial-rng.pt',weights_only=False))
        sys.setprofile(sizes)
        try:lesson(observer)
        finally:sys.setprofile(None)
    result=sizes.report();result['equal_batches']=compare_states(args.output,args.compare)
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('equal batches',result['equal_batches'])

if __name__=='__main__':main()
