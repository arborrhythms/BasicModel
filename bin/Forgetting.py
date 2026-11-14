"""Activity passes over the single retained history and its relation table."""
from dataclasses import dataclass, fields, is_dataclass, replace
import math
import warnings
import torch


@dataclass(frozen=True)
class ForgettingConfig:
    enabled: bool = True
    high_water: float = .9
    low_water: float = .7
    min_interval: int = 256
    at_epoch: bool = False

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.high_water,self.low_water)):
            raise ValueError('forgetting water marks must be finite')
        if not 0 <= self.low_water < self.high_water <= 1 or self.min_interval < 0:
            raise ValueError('invalid forgetting water marks or interval')

    @classmethod
    def from_xml(cls):
        from util import TheXMLConfig
        defaults=cls()
        names=dict(enabled='forgetting',high_water='forgetHighWater',low_water='forgetLowWater',
                   min_interval='forgetMinInterval',at_epoch='forgetAtEpoch')
        values={}
        for item in fields(cls):
            default=getattr(defaults,item.name)
            value=TheXMLConfig.get('architecture.'+names[item.name],default=default)
            if isinstance(default,bool) and isinstance(value,str):
                if value.lower() not in ('true','false','1','0'):
                    raise ValueError('forgetting boolean must be true or false')
                value=value.lower() in ('true','1')
            values[item.name]=type(default)(value)
        return cls(**values)


def _live_occurrences(value, namespace):
    pending, found, seen = [value], set(), set()
    while pending:
        item=pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if is_dataclass(item) and not isinstance(item,type):
            pending.extend(getattr(item,field.name) for field in fields(item))
        elif isinstance(item,dict):
            pending.extend(item.values())
        elif isinstance(item,(list,tuple)):
            if len(item)==3 and item[:2]==('ltm',namespace) and type(item[2]) is int:
                found.add(tuple(item))
            else:
                pending.extend(item)
    return found


@torch.no_grad()
def forget_parts(parts, config, *, retained=(), now=None):
    """Decay once; remove the least active unreferenced, nonpermanent pairs."""
    from MemoryActivity import decay
    from Memory import RELATION
    rows=parts.indices().tolist()
    decay(parts,rows)
    if len(rows) < math.ceil(config.high_water*parts.capacity):
        return None
    target=math.floor(config.low_water*parts.capacity)
    protected={row for row in rows if float(parts.activity[row])==1.}
    protected.update(int(ref[1]) for ref in retained
        if ref[:1]==('sym',) and parts.contains(ref[1]))
    protected.update(value for kind,value in parts.protected_references() if kind==RELATION)
    alive=set(rows)
    for row in sorted(rows,key=lambda row:(float(parts.activity[row]),row)):
        if len(alive)<=target:
            break
        if row not in protected:
            alive.remove(row)
    parts.retain(sorted(alive))
    return dict(before=len(rows),after=len(alive),target=target,protected=len(protected),
                target_reached=len(alive)<=target)


@torch.no_grad()
def forget(store, config=None, *, document_boundary=False, epoch_boundary=False,
           open_sentence=False, stateless=False, retained_occurrences=()):
    config=config or ForgettingConfig()
    if (not config.enabled or stateless or open_sentence or torch.compiler.is_compiling()
            or not (document_boundary or epoch_boundary)):
        return None
    pressure=(len(store)>=math.ceil(config.high_water*store.capacity)
              or bool(store.forget_pending))
    parts_pressure=len(store.parts)>=math.ceil(config.high_water*store.parts.capacity)
    if not (pressure or parts_pressure or bool(store.forget_pending) or (epoch_boundary and config.at_epoch)):
        return None
    if int(store.forget_passes) and int(store.forget_sentences-store.forget_last_sentence)<config.min_interval:
        return None
    from Memory import OCCURRENCE
    from MemoryActivity import decay
    # Existing reference integrity is kept: a child of a retained row cannot
    # vanish underneath it. A later pass can release a former dependency.
    references=set(retained_occurrences)
    store.last_part_forgetting_report=forget_parts(store.parts,config,retained=references)
    decay(store,range(len(store)))
    target=math.floor(config.low_water*store.capacity)
    n=len(store)
    protected={row for row in range(n) if float(store.activity[row])==1.}
    references.update(('ltm',store.address_domain,value)
        for kind,value in store.parts.protected_references() if kind==OCCURRENCE)
    references.update(ref for row in range(n) if (ref:=store.referent_of(row)) is not None)
    for reference in references:
        if reference[:2]==('ltm',store.address_domain):
            index=store.location(reference)
            if index is None:
                raise ValueError('forgetting retained reference is unavailable')
            protected.add(index)
    alive=set(range(n))
    cutoff=None
    if pressure:
        for row in sorted(alive,key=lambda row:(float(store.activity[row]),store.row_name(row))):
            if len(alive)<=target:
                break
            if row not in protected:
                cutoff=float(store.activity[row]);alive.remove(row)
    store.compact_rows(sorted(alive))
    store.forget_passes.add_(1)
    store.forget_pending.fill_(len(store)>target if pressure else False)
    store.forget_last_sentence.copy_(store.forget_sentences)
    report=dict(before=n,after=len(store),target=target,cutoff=cutoff,
                protected=len(protected),target_reached=len(store)<=target if pressure else True)
    store.last_forgetting_report=report
    from util import TheMessage
    TheMessage(f'[LTM forgetting] {report}')
    if not report['target_reached']:
        warnings.warn('activity pass cannot reach low water while preserving retained references',RuntimeWarning)
    return report


def model_boundary(model, *, document_boundary=False, epoch_boundary=False, config=None):
    store=getattr(getattr(model,'symbolSpace',None),'ltm_store',None)
    if store is None:
        return None
    roots=_live_occurrences(getattr(model,'_last_understanding',None),store.address_domain)
    memory=getattr(model.symbolSpace,'what_memory',None)
    if memory is not None:
        roots.update(memory.retained_ltm_occurrences(store.address_domain))
    discourse=getattr(model.symbolSpace,'expectation',None)
    if discourse is not None:
        for chain in getattr(discourse,'_inter_context_occurrences',()):
            roots.update(ref for ref in chain if ref is not None)
        for pending in getattr(discourse,'_inter_last_meaning',()):
            if pending is not None:
                roots.update(getattr(pending,'source_occurrences',()))
    result=forget(store,config or ForgettingConfig.from_xml(),
        document_boundary=document_boundary,epoch_boundary=epoch_boundary,
        open_sentence=getattr(model,'_open_sentence_slot',None) is not None,
        retained_occurrences=roots)
    if result is not None:
        truth=getattr(model.symbolSpace,'truth_layer',None)
        if truth is not None:
            truth.sync_from_ltm()
    return result


def serving_pass(model):
    """A serving write under pressure runs the same activity pass first."""
    config=replace(ForgettingConfig.from_xml(),min_interval=0)
    return model_boundary(model,document_boundary=True,config=config)
