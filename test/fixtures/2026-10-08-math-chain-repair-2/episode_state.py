"""Development footprint observer; all reads are detached, no learner input."""
import dataclasses,hashlib
from collections import deque
from collections.abc import Mapping
import torch


def value_digest(value, seen=None):
    seen=set() if seen is None else seen
    if torch.is_tensor(value):
        data=value.detach().cpu()
        if data.layout != torch.strided:
            data=data.to_dense()
        data=data.contiguous()
        return ('tensor',str(data.dtype),tuple(data.shape),hashlib.sha256(
            data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if value is None or isinstance(value,(str,int,float,bool,bytes)):
        return value
    if id(value) in seen:
        return ('recursive',type(value).__name__)
    seen.add(id(value))
    try:
        if isinstance(value,Mapping):
            return tuple((repr(k),value_digest(v,seen)) for k,v in value.items())
        if isinstance(value,(tuple,list,deque)):
            return tuple(value_digest(v,seen) for v in value)
        if dataclasses.is_dataclass(value):
            return tuple((f.name,value_digest(getattr(value,f.name),seen)) for f in dataclasses.fields(value))
        return ('object',type(value).__name__,id(value))
    finally:
        seen.remove(id(value))


def snapshot(model):
    state={}
    for module_name,module in model.named_modules():
        for key,value in vars(module).items():
            if key in ('_modules','_parameters','_buffers'):
                continue
            state[module_name+'.'+key]=value_digest(value)
        for key,value in module._buffers.items():
            state[module_name+'.buffer.'+key]=value_digest(value)
        for key,value in module._parameters.items():
            state[module_name+'.parameter.'+key]=value_digest(value)
    memory=model._what_memory()
    if memory is not None:
        for key,value in vars(memory).items():
            state['what_memory.'+key]=value_digest(value)
    return state


def difference(before,after):
    return [key for key in sorted(set(before)|set(after)) if before.get(key)!=after.get(key)]
