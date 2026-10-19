"""H3: measure the retained prior at declared bank/row capacities, no training."""
import json,sys,time,tracemalloc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'bin'),str(ROOT/'test')]
import torch
from meaning_fixtures import symbol_prediction

def deep_size(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value));size=sys.getsizeof(value)
    if isinstance(value,dict):return size+sum(deep_size(k,seen)+deep_size(v,seen) for k,v in value.items())
    if isinstance(value,(tuple,list)):return size+sum(deep_size(v,seen) for v in value)
    return size

def measure():
    torch.set_num_threads(1)
    n,width,capacity=1024,1032,1024
    keys=torch.zeros(n,width);keys[torch.arange(n),torch.arange(n)]=1
    prior=symbol_prediction(keys,torch.full((n,2),.5))
    target=torch.zeros(n,2);target[torch.tensor([10,20,30]),0]=1
    roles=keys[torch.tensor([10,20,30])]
    rows=[]
    for name,kw in [('full_prior',{}),('retained_row',dict(observed=roles,occupied=torch.ones(3,dtype=torch.bool),activation=target))]:
        tracemalloc.start();start=time.perf_counter()
        bindings=prior.retained_bindings(**kw)
        packed=time.perf_counter()-start
        start=time.perf_counter();serialized=json.dumps(bindings)
        encoding=time.perf_counter()-start
        peak=tracemalloc.get_traced_memory()[1];tracemalloc.stop()
        size=deep_size(bindings)
        rows.append(dict(mode=name,keys_retained=sum(e[2] is not None for e in bindings['_expected_symbols']),
            python_bytes=size,json_bytes=len(serialized.encode()),peak_python_allocation=peak,
            packing_seconds=packed,json_seconds=encoding,projected_bytes_at_ltm_capacity=size*capacity))
    return dict(bank_size=n,key_width=width,ltm_capacity=capacity,observed_symbols=3,measurements=rows,
        limitation='Metadata bound only. The retained reader materializes one dense bank view transiently (4 MiB here); no bank is reconstructed from later mutable model weights. Unpaired future priors may retain all keys until an observation is attached.')

if __name__=='__main__':
    output=measure();Path(__file__).with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output))
