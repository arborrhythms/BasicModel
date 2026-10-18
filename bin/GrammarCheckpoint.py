"""One-way checkpoint alignment by declared grammar identities.

No retired operator can execute here. Old checkpoints without a catalogue
require an explicitly supplied catalogue from their archived source.
"""
import copy
import torch


def catalog(model):
    from Language import TheGrammar, SyntacticLayer, OperationSelectionLayer, compute_role_vocabulary
    return dict(rule_keys=list(TheGrammar.rule_table.values()),
        roles=compute_role_vocabulary(TheGrammar)[0], categories=list(TheGrammar.categories),
        syntactic={name:list(module._by_name) for name,module in model.named_modules()
                   if isinstance(module,SyntacticLayer)},
        operation={name:dict(binary=module.op_names,unary=module.unary_names)
                   for name,module in model.named_modules() if isinstance(module,OperationSelectionLayer)})


def _identities(names):
    # Several semantic faces can share a numerical host (lower). Their
    # stable declaration order distinguishes those existing rows.
    return [(name, names[:i].count(name)) for i,name in enumerate(names)]


def _mapping(old, new):
    return tuple(old.index(name) if name in old else -1 for name in new)


def align(model, state, old):
    """Return migrated tensors, explicit drift, and optimizer transformations."""
    live, new = model.state_dict(), catalog(model)
    result, drift, transforms = dict(state), {}, {}
    for key in tuple(result):
        if key.endswith(('.timestamp','._next_ts')):
            del result[key]
            drift[key] = dict(action='dropped retired clock')
    for key,value in live.items():
        if key.endswith('.stream') and key not in result:
            result[key] = value.detach().clone()
            drift[key] = dict(action='initialized empty stream history; checkpoint has no stream witnesses')
    if old is None:
        return result, drift, transforms

    # ModuleList positions are storage locations, never operator identities.
    for prefix,names in old['syntactic'].items():
        current = new['syntactic'].get(prefix, ())
        pending = {}
        for index,name in enumerate(names):
            source = f'{prefix}.layers.{index}.'
            for key in [key for key in result if key.startswith(source)]:
                value = result.pop(key)
                if name not in current:
                    drift[key] = dict(action='dropped retired operator')
                    transforms[key] = None
                    continue
                target = f'{prefix}.layers.{current.index(name)}.'+key[len(source):]
                pending[target] = value
                if key != target:
                    drift[key] = dict(action='renamed',target=target)
                    transforms[key] = (target,None,None)
        result.update(pending)

    rules = _mapping(old['rule_keys'],new['rule_keys'])
    roles = _mapping(old['roles'],new['roles'])
    categories = _mapping(old['categories'],new['categories'])
    operations = {}
    for prefix,new_ops in new['operation'].items():
        if prefix not in old['operation']:
            continue
        old_ops = old['operation'][prefix]
        binary = _mapping(_identities(old_ops['binary']),_identities(new_ops['binary']))
        unary = _mapping(_identities(old_ops['unary']),_identities(new_ops['unary']))
        tools = (0,)+tuple(-1 if i<0 else 1+i for i in binary)+tuple(
            -1 if i<0 else 1+len(old_ops['binary'])+i for i in unary)+tuple(
            range(1+len(old_ops['binary'])+len(old_ops['unary']),
                  7+len(old_ops['binary'])+len(old_ops['unary'])))
        owner = model.get_submodule(prefix)
        for alias,module in model.named_modules(remove_duplicate=False):
            if module is owner:
                operations[alias+'.'] = (binary,unary,tools)

    for key,value in tuple(result.items()):
        if key not in live or not torch.is_tensor(value):
            continue
        axis, indices = None, None
        if key.endswith('routing_proj.weight'):
            axis,indices=1,rules
        elif '.rule_predictor.2.' in key:
            axis,indices=0,rules
        elif key.endswith('.category_embedding.weight'):
            axis,indices=0,categories
            indices += tuple(range(len(indices),live[key].shape[0]))
        elif '.operation_layer.' in key:
            matches = [axes for prefix,axes in operations.items() if key.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f'grammar catalogue has no unique operation owner: {key}')
            binary,unary,tools = matches[0]
            if key.endswith('.reduce_anchor'):axis,indices=0,binary
            elif key.endswith('.apply_anchor'):axis,indices=0,unary
            elif key.endswith(('.tool_embedding','.what_projection.weight')):axis,indices=0,tools
            elif key.endswith(('.mlp.0.weight','.operand_order.weight','.copy_order.weight')):
                suffix = 8 if key.endswith('.mlp.0.weight') else 0
                prefix = value.shape[1]-len(old['roles'])-suffix
                axis=1
                indices=tuple(range(prefix))+tuple(-1 if i<0 else prefix+i for i in roles)+tuple(
                    range(value.shape[1]-suffix,value.shape[1]))
        if axis is None or (value.shape==live[key].shape and indices==tuple(range(value.shape[axis]))):
            continue
        if len(indices)!=live[key].shape[axis] or any(i>=value.shape[axis] for i in indices):
            raise ValueError(f'grammar catalogue does not describe checkpoint tensor: {key}')
        target = live[key].detach().to(value).clone()
        if axis==1:
            target.zero_()  # new context coordinates have no learned contribution
        for dst,src in enumerate(indices):
            if src>=0:target.select(axis,dst).copy_(value.select(axis,src))
        result[key]=target
        drift[key]=dict(action='aligned grammar axis',axis=axis,retained=sum(i>=0 for i in indices),
                        initialized=sum(i<0 for i in indices))
        transforms[key]=(key,axis,indices)
    return result,drift,transforms


def align_optimizer(state, manifest, transforms, tensors):
    """Apply the same name/axis map to moments; new coordinates start at zero."""
    if state is None or manifest is None or not transforms:
        return state,manifest
    from checkpoint_migrations import _optimizer_state_leaves
    state,manifest=copy.deepcopy(state),copy.deepcopy(manifest)
    for leaf,metadata in zip(_optimizer_state_leaves(state),manifest['leaves'],strict=True):
        for group,entries in zip(leaf['param_groups'],metadata['param_groups'],strict=True):
            for pid,entry in zip(group['params'],entries,strict=True):
                name=entry['name']
                if name not in transforms:continue
                transform=transforms[name]
                if transform is None:
                    leaf['state'].pop(pid,None)
                    continue
                target,axis,indices=transform
                entry['name']=target
                if axis is None:continue
                entry['shape']=list(tensors[target].shape)
                for field,value in tuple(leaf['state'].get(pid,{}).items()):
                    if not torch.is_tensor(value) or value.ndim==0 or value.numel()==1:continue
                    replacement=value.new_zeros(tensors[target].shape)
                    for dst,src in enumerate(indices):
                        if src>=0:replacement.select(axis,dst).copy_(value.select(axis,src))
                    leaf['state'][pid][field]=replacement
    return state,manifest
