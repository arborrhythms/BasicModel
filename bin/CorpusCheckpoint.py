"""Continuation state at a completed corpus brick, outside the model weights.

These are observations and ledgers, never a saved computation graph. The
ordinary structural checkpoint already owns concepts, retained rows and
thought history. This companion preserves the open presentation's readers.
"""
from collections import deque
from dataclasses import fields, is_dataclass, replace
import copy

import torch


CONTEXT = ('_stm_end_states', '_inter_context', '_inter_context_lanes',
           '_inter_context_ids', '_inter_context_occurrences',
           '_inter_last_meaning', '_expectation_documents')
SURFACES = ('_priming_boosts', '_priming_seen')
STM_META = ('_active_batch', '_scope_columns', '_whole_columns', '_live_max_depth_host')


def stm_owners(model):
    from Layers import ShortTermMemory
    return {name: module for name, module in model.named_modules()
            if isinstance(module, ShortTermMemory)}


def stm_fields():
    from STMChunks import FIELDS
    return tuple('_live_'+name for name in (*FIELDS, 'clause_state', 'wholes',
                                           'last_unary', 'unary_history_depth'))


def detached(value, device='cpu'):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: detached(getattr(value, field.name), device)
                                 for field in fields(value)})
    if isinstance(value, dict):
        return {copy.deepcopy(key): detached(item, device) for key, item in value.items()}
    if isinstance(value, deque):
        return deque((detached(item, device) for item in value), maxlen=value.maxlen)
    if isinstance(value, (tuple, list)):
        return type(value)(detached(item, device) for item in value)
    return copy.deepcopy(value)


def collect(model):
    if getattr(model, '_sentence_run_active', False):
        raise RuntimeError('corpus checkpoint requires a completed sentence transaction')
    surfaces = {name: {key: detached(getattr(module, key)) for key in SURFACES}
                for name, module in model.named_modules()
                if all(torch.is_tensor(getattr(module, key, None)) for key in SURFACES)}
    expectation = getattr(getattr(model, 'symbolSpace', None), 'expectation', None)
    context = None if expectation is None else dict(batch=expectation._batch,
        fields={key: detached(getattr(expectation, key)) for key in CONTEXT})
    return dict(version=1, surfaces=surfaces, expectation=context,
        percept_expectation=detached(model.percept_expectation.continuation()),
        stm={name: dict(fields={key: detached(getattr(module, key)) for key in stm_fields()},
                        metadata={key: int(getattr(module, key)) for key in STM_META})
             for name, module in stm_owners(model).items()},
        mtm=[space.mtm.state() for space in model.conceptualSpaces],
        document_covers=detached(getattr(model, '_document_covers', {})),
        curriculum_batches=int(getattr(model, '_curriculum_batches', 0)))


@torch.no_grad()
def restore(model, state):
    if not isinstance(state, dict) or state.get('version') != 1:
        raise ValueError('corpus checkpoint requires continuation state version 1')
    modules = dict(model.named_modules())
    expected = {name for name, module in modules.items()
                if all(torch.is_tensor(getattr(module, key, None)) for key in SURFACES)}
    if set(state.get('surfaces', ())) != expected:
        raise ValueError('corpus priming owners differ from construction')
    writes = []
    for name, values in state['surfaces'].items():
        if set(values) != set(SURFACES):
            raise ValueError('corpus checkpoint requires both priming ledgers')
        for key, value in values.items():
            target = getattr(modules[name], key)
            if (not torch.is_tensor(value) or value.shape != target.shape
                    or value.dtype != target.dtype or not torch.isfinite(value).all()
                    or (value < 0).any()):
                raise ValueError('corpus priming ledger differs from construction')
            writes.append((target, value))
    spaces = list(model.conceptualSpaces)
    if len(state.get('mtm', ())) != len(spaces):
        raise ValueError('corpus identity owners differ from construction')
    stms = stm_owners(model)
    if set(state.get('stm', ())) != set(stms):
        raise ValueError('corpus working-memory owners differ from construction')
    for name, module in stms.items():
        saved = state['stm'][name]
        meta, values = saved.get('metadata', {}), saved.get('fields', {})
        if (set(meta) != set(STM_META) or any(type(v) is not int for v in meta.values())
                or not 1 <= meta['_active_batch'] <= module._init_batch
                or meta['_scope_columns'] not in (2, 14)
                or meta['_whole_columns'] not in (3, 5)
                or not 0 <= meta['_live_max_depth_host'] <= module.capacity
                or set(values) != set(stm_fields())):
            raise ValueError('corpus working-memory metadata differs from construction')
        for key, value in values.items():
            target = getattr(module, key)
            if (not torch.is_tensor(value) or value.shape != target.shape
                    or value.dtype != target.dtype or not torch.isfinite(value).all()):
                raise ValueError('corpus working-memory slab differs from construction')
            writes.append((target, value))
    expectation = getattr(getattr(model, 'symbolSpace', None), 'expectation', None)
    context = state.get('expectation')
    if (expectation is None) != (context is None):
        raise ValueError('corpus expectation owner differs from construction')
    if context is not None:
        batch = context.get('batch')
        if (type(batch) is not int or batch < 1 or batch > spaces[0].nStreams
                or set(context.get('fields', ())) != set(CONTEXT)
                or any(len(value) != batch for value in context['fields'].values())):
            raise ValueError('corpus expectation streams differ from construction')
        for b in range(batch):
            sizes = [len(context['fields'][name][b]) for name in
                     ('_inter_context', '_inter_context_lanes', '_inter_context_ids',
                      '_inter_context_occurrences')]
            if len(set(sizes)) != 1 or sizes[0] > expectation._inter_chain_window:
                raise ValueError('corpus expectation contexts are misaligned')
    for target, value in writes:
        target.copy_(value.to(target.device))
    for name, module in stms.items():
        for key, value in state['stm'][name]['metadata'].items():
            setattr(module, key, value)
    for space, saved in zip(spaces, state['mtm']):
        space.mtm.restore(saved, device=space.similarity_codebook.W.device)
    if context is not None:
        expectation.ensure_batch(context['batch'])
        device = next(expectation.parameters()).device
        for name, value in context['fields'].items():
            setattr(expectation, name, detached(value, device))
    if 'percept_expectation' not in state:
        raise ValueError('corpus checkpoint lacks percept expectation optimizer state')
    model.percept_expectation.restore_continuation(state['percept_expectation'])
    model._document_covers = detached(state['document_covers'])
    model._curriculum_batches = int(state['curriculum_batches'])
