"""Development footprint observer; all reads are detached, no learner input."""
import dataclasses,hashlib
from collections import deque
from collections.abc import Mapping
import torch


def value_digest(value, seen=None, cache=None):
    seen=set() if seen is None else seen
    cache={} if cache is None else cache
    if id(value) in cache:
        return cache[id(value)]
    if torch.is_tensor(value):
        data=value.detach().cpu()
        if data.layout != torch.strided:
            # A sparse dictionary's logical capacity can be enormous. Audit
            # its canonical occupied storage without materializing that space.
            data=data.to_sparse_coo().coalesce()
            indices,values=data.indices(),data.values()
            result=('sparse_tensor',str(data.dtype),tuple(data.shape),
                    value_digest(indices),value_digest(values))
            cache[id(value)]=result
            return result
        data=data.contiguous()
        result = ('tensor',str(data.dtype),tuple(data.shape),hashlib.sha256(
            data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
        cache[id(value)] = result
        return result
    if value is None or isinstance(value,(str,int,float,bool,bytes)):
        return value
    if id(value) in seen:
        return ('recursive',type(value).__name__)
    seen.add(id(value))
    try:
        if isinstance(value,Mapping):
            return tuple((repr(k),value_digest(v,seen,cache)) for k,v in value.items())
        if isinstance(value,(tuple,list,deque)):
            return tuple(value_digest(v,seen,cache) for v in value)
        if dataclasses.is_dataclass(value):
            return tuple((f.name,value_digest(getattr(value,f.name),seen,cache)) for f in dataclasses.fields(value))
        return ('object',type(value).__name__,id(value))
    finally:
        seen.remove(id(value))


def snapshot(model):
    state={}
    cache={}
    for module_name,module in model.named_modules():
        for key,value in vars(module).items():
            if key in ('_modules','_parameters','_buffers'):
                continue
            state[module_name+'.'+key]=value_digest(value,cache=cache)
        for key,value in module._buffers.items():
            state[module_name+'.buffer.'+key]=value_digest(value,cache=cache)
        for key,value in module._parameters.items():
            state[module_name+'.parameter.'+key]=value_digest(value,cache=cache)
    memory=model._what_memory()
    if memory is not None:
        for key,value in vars(memory).items():
            state['what_memory.'+key]=value_digest(value,cache=cache)
    return state


def difference(before,after):
    return [key for key in sorted(set(before)|set(after)) if before.get(key)!=after.get(key)]


# Keep the raw comparator available even when an archived measurement plugin
# wraps ``difference``. The reusable certificate records the same approved
# cache/chunk classification itself, including every raw before/after digest.
raw_difference = difference


def observed_difference(model, before, after):
    """Apply the landed observer ports without omitting any raw state change."""
    from MereologicalCodes import MereologicalCodes
    from Layers import ShortTermMemory
    import STMChunks

    cache_fields, chunk_fields, chain_fields = set(), set(), {}
    store = getattr(getattr(model, 'symbolSpace', None), 'ltm_store', None)
    chains = set() if store is None else {repr(int(k)) for k in store.parts.indices().tolist()}
    for name, module in model.named_modules():
        if isinstance(module, MereologicalCodes):
            assert '_form_valid' in module._non_persistent_buffers_set
            assert not module._form_valid.requires_grad and isinstance(module.form_rows_derived, int)
            cache_fields.update((name + '.buffer._form_valid', name + '.form_rows_derived'))
        if isinstance(module, ShortTermMemory):
            assert len(module.state) == 13 and module.capacity == 8
            assert all(value.shape[0] <= module._init_batch for value in module.state)
            assert all(value.shape[1] == 8 for i, value in enumerate(module.state)
                       if i != STMChunks.DEPTH)
            chunk_fields.update(name + '._live_' + field for field in STMChunks.FIELDS)
            chunk_fields.add(name + '._live_max_depth_host')
        if hasattr(module, '_constituents'):
            chain_fields[name + '._constituents'] = chains
    changed = raw_difference(before, after)

    def substantive(value, key):
        return tuple((row, entries) for row, entries in value
                     if entries or row not in chain_fields[key])

    registrations = [key for key in changed if key in chain_fields
                     and substantive(before[key], key) == substantive(after[key], key)]
    transient = [key for key in changed
                 if key in cache_fields or key in chunk_fields or key in registrations]
    return dict(changed=changed, transient=transient,
                empty_chain_registrations=registrations,
                comparison=[key for key in changed if key not in transient],
                cache_values={key: dict(before=before.get(key), after=after.get(key))
                              for key in transient})
