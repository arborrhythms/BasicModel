"""One-way import of the preceding tables into time-named idea rows."""
from dataclasses import fields, is_dataclass, replace
import warnings
import torch


def prepare(model, state):
    """Validate every legacy store before any destination module is loaded."""
    from Memory import LTM0
    stores=[(name+'.' if name else '',module) for name,module in model.named_modules(remove_duplicate=False)
            if isinstance(module,LTM0)]
    for prefix,store in stores:
        if prefix+'slots' in state:
            store.validate_checkpoint(state,prefix)
    for prefix,store in stores:
        if prefix+'slots' in state:
            store.prepare_checkpoint(state,prefix)


def finish_legacy(model, extras):
    """Move retained identities to referent, pending bindings to MTM, and discard provenance."""
    from Memory import LTM0,OCCURRENCE
    modules=dict(model.named_modules())
    stores=[module for module in modules.values() if isinstance(module,LTM0)]
    old={}
    for store in stores:
        for key,tick in store.__dict__.get('_legacy_times',{}).items():
            value=('ltm',store.address_domain,int(tick))
            if key in old and old[key]!=value:
                raise ValueError('ambiguous legacy occurrence owner')
            old[key]=value

    def remap(value, *, typed=False):
        if isinstance(value,dict) and value.get('namespace') == 'address-v1' and 'rows' in value:
            value=dict(value,namespace='time-v1')
        if isinstance(value,tuple) and len(value)==3 and value[:2]==('ltm','address-v1'):
            if value[2] not in old:
                raise ValueError('legacy occurrence reference has no retained row')
            return old[value[2]]
        if (isinstance(value,tuple) and len(value)==2
                and (value[0]=='occ' or (typed and value[0]==OCCURRENCE))):
            if value[1] not in old:
                raise ValueError('legacy member reference has no retained row')
            return value[0],old[value[1]][2]
        if isinstance(value,dict):
            return {remap(key,typed=typed):remap(item,typed=typed) for key,item in value.items()}
        if isinstance(value,(tuple,list,set,frozenset)):
            return type(value)(remap(item,typed=typed) for item in value)
        if is_dataclass(value) and not isinstance(value,type):
            return replace(value,**{field.name:remap(getattr(value,field.name))
                                   for field in fields(value) if field.init})
        return value

    for name,saved in (extras.get('truth_semantics') or {}).items():
        store=modules.get(name)
        if not isinstance(store,LTM0) or saved.get('namespace')!='address-v1':
            raise ValueError('legacy semantic records have no matching idea table')
        owner=store.owner()
        for record in saved.get('records',()):
            reference=remap(('ltm','address-v1',record['id']))
            row=store.location(reference)
            if row is None:
                raise ValueError('legacy semantic record has no retained idea')
            marks=dict(remap(dict(record.get('context',{}).get('bindings',()))))
            identity=marks.get('_identity')
            if identity is not None and identity[0] in ('sym','ltm'):
                store.referent[row]=store.referent.new_tensor(store._reference_value(identity))
            if owner is not None:
                owner.mtm.hold_pending(reference,replace(store.raw_meaning_of(row),bindings=marks))
        warnings.warn('checkpoint: moved selected referents and pending bindings; discarded retired semantic provenance')
    if extras.get('epoch_rows'):
        warnings.warn('checkpoint: discarded retired presentation row history')
    seen=set()
    owners=set()
    for store in stores:
        mapping=store.__dict__.get('_legacy_times')
        if mapping is None:
            continue
        parts=store.parts
        if id(parts) not in seen:
            seen.add(id(parts))
            slots=parts.ref_kinds.eq(OCCURRENCE).nonzero().tolist()
            for row,end in slots:
                key=int(parts.refs[row,end])
                if key not in mapping:
                    raise ValueError('legacy relation names an unavailable occurrence')
                parts.refs[row,end]=mapping[key]
            parts.clear_caches();parts.rebuild_caches()
        owner=store.owner()
        if owner is not None and id(owner) not in owners:
            owners.add(id(owner))
            owner.mtm.restore(remap(owner.mtm.state()),device=owner.similarity_codebook.W.device)
            alloc=owner._concept_allocator
            for layer in alloc._layers.values():
                records=remap(layer._constituents)
                layer._constituents.clear();layer._constituents.update(records)
            for name in ('relate_idx','singleton_members','singleton_rows'):
                current=getattr(alloc,name)
                records=remap(current,typed=True)
                current.clear();current.update(records)
            components=owner.components
            # These are retained occurrence witnesses, not concept IDs.
            components._innovations={old[key][2]:value for key,value in components._innovations.items()}
            components.verbs._witnesses=remap(components.verbs._witnesses)
            components.verbs._pending=remap(components.verbs._pending)
            components._observation_revision=None
        store.__dict__.pop('_legacy_times',None)
    return remap
