"""One-way checkpoint alignment by declared grammar identities.

No retired operator can execute here. Old checkpoints without a catalogue
require an explicitly supplied catalogue from their archived source.
"""
import copy
import torch


def catalog(model):
    from Language import TheGrammar, SyntacticLayer, OperationSelectionLayer
    return dict(rule_keys=list(TheGrammar.rule_table.values()),
        roles=[], chooser_role_context=0, occupation_schema=1, categories=list(TheGrammar.categories),
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
    from IdentityRows import migrate_checkpoint
    migrate_checkpoint(model, result, live, drift, transforms)
    from Layers import LTM0
    from RowTime import migrate
    for name, module in model.named_modules(remove_duplicate=False):
        if isinstance(module, LTM0) and name+'.slots' in result:
            if migrate(module, result, name+'.'):
                drift[name+'.when'] = dict(action='historical model time unknown; source-position phasor retired')
    from Language import MarkerUnaryLayer
    identities = {name+'.marker' for name, module in model.named_modules(remove_duplicate=False)
                  if isinstance(module, MarkerUnaryLayer)
                  and getattr(module.base, 'left_identity', None) is not None}

    def retire_markers():
        for key in identities.intersection(result):
            del result[key]
            drift[key] = dict(action='replaced free unary marker with operator identity')
            transforms[key] = None
            for source, transform in tuple(transforms.items()):
                if transform is not None and transform[0] == key:
                    transforms[source] = None
    for key in tuple(result):
        if any(part in ('_category_vq', '_category_role') for part in key.split('.')):
            del result[key]
            drift[key] = dict(action='dropped retired category codebook')
            transforms[key] = None
            continue
        if key.endswith(('.timestamp','._next_ts')):
            del result[key]
            drift[key] = dict(action='dropped retired clock')
    for key,value in live.items():
        if key.endswith('.stream') and key not in result:
            result[key] = value.detach().clone()
            drift[key] = dict(action='initialized empty stream history; checkpoint has no stream witnesses')
    for key in live:
        if not key.endswith('.parts_of_speech._extra_state'):
            continue
        if key in result:
            from IndependentComponents import _pack_state, _unpack_state
            state = _unpack_state(result[key])
            owner = model.get_submodule(key[:-len('._extra_state')])
            if state['version'] != 1:
                raise ValueError('unknown occupation checkpoint schema')
            if tuple(state['arguments']) != owner.arguments:
                if old is None:
                    raise ValueError('occupation migration requires the saved grammar catalogue')
                counts = {word: {argument: count for argument, count in values.items()
                                 if argument in owner.index}
                          for word, values in state['counts'].items()}
                result[key] = _pack_state(dict(state, arguments=owner.arguments, counts=counts))
                drift[key] = dict(action='aligned declared formal arguments',
                                  retired=sorted(set(state['arguments']) - set(owner.arguments)))
        else:
            if old is not None and old.get('occupation_schema') == 1:
                raise ValueError('checkpoint is missing its occupation state')
            from PartsOfSpeech import PartsOfSpeech
            owner = model.get_submodule(key[:-len('._extra_state')])
            result[key] = PartsOfSpeech(owner.arguments).get_extra_state()
            drift[key] = dict(action='initialized empty occupation counts; old categories are not evidence')
    if old is None:
        retire_markers()
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
                # Operation modules have positional storage too. Remap all
                # aliases from the same declared catalogue before inspecting
                # tensor shapes; a shifted operator must not inherit its
                # predecessor's parameters or optimizer moments.
                for kind, field in (('binary', 'ops'), ('unary', 'unary_ops')):
                    previous = _identities(old_ops[kind])
                    current = _identities(new_ops[kind])
                    head = f'{alias}.{field}.'
                    pending = {}
                    for key in [key for key in result if key.startswith(head)]:
                        position, suffix = key[len(head):].split('.', 1)
                        index = int(position)
                        if index >= len(previous):
                            raise ValueError(f'grammar catalogue does not describe operator: {key}')
                        value = result.pop(key)
                        identity = previous[index]
                        if identity not in current:
                            drift[key] = dict(action='dropped retired operator')
                            transforms[key] = None
                            continue
                        target = f'{head}{current.index(identity)}.{suffix}'
                        pending[target] = value
                        if key != target:
                            drift[key] = dict(action='renamed', target=target)
                            transforms[key] = (target, None, None)
                    result.update(pending)

    retire_markers()

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
        elif any(key.startswith(prefix) for prefix in operations):
            matches = [axes for prefix,axes in operations.items() if key.startswith(prefix)]
            if len(matches) != 1:
                raise ValueError(f'grammar catalogue has no unique operation owner: {key}')
            binary,unary,tools = matches[0]
            if key.endswith('.reduce_anchor'):axis,indices=0,binary
            elif key.endswith('.apply_anchor'):axis,indices=0,unary
            elif key.endswith(('.tool_embedding','.what_projection.weight')):axis,indices=0,tools
            elif key.endswith(('.mlp.0.weight','.operand_order.weight','.copy_order.weight')):
                suffix = 8 if key.endswith('.mlp.0.weight') else 0
                prefix = value.shape[1]-old.get('chooser_role_context', len(old['roles']))-suffix
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
