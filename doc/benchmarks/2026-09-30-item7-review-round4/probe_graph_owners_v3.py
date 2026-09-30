"""Observe the unchanged graph-release gate by model owner and saved-value owner."""
from collections import defaultdict
from functools import wraps
import json
import os
from pathlib import Path
import traceback
import weakref

import torch
from torch._subclasses.fake_tensor import is_fake
import Models
import SentenceCompose
from bounded_tests import ProcessTree
import test_word_store as original


def test_graph_release_memory_owners(monkeypatch):
    destination = Path(os.environ['ITEM7_OWNER_LOG'])
    live = weakref.WeakKeyDictionary()
    owners = {}
    process = ProcessTree(os.getpid())
    sequence = 0
    saved_init = SentenceCompose._SavedValue.__init__

    def identify(model):
        owners.clear()
        for name, value in (*model.named_parameters(), *model.named_buffers()):
            if value.layout == torch.strided and not is_fake(value):
                owners.setdefault(value.untyped_storage()._cdata, name)

    def observe_saved(self, value, storage, copy):
        saved_init(self, value, storage, copy)
        if is_fake(value) or value.layout != torch.strided:
            return
        source = None if storage is None else storage._cdata
        owner = owners.get(source)
        if owner is None:
            frames = traceback.extract_stack(limit=24)
            selected = [f for f in frames if '/bin/' in f.filename and f.name != 'pack']
            owner = ('activation:' + '/'.join(f'{Path(f.filename).name}:{f.name}' for f in selected[-3:]))
        live[self] = dict(owner=owner, copied=copy, shape=list(value.shape))

    def snapshot(model, phase, sid=None):
        nonlocal sequence
        sequence += 1
        saved, counts, seen = defaultdict(int), defaultdict(int), set()
        for value, metadata in tuple(live.items()):
            storage = value.value.untyped_storage()
            if storage._cdata in seen:
                continue
            seen.add(storage._cdata)
            key = ('copy:' if metadata['copied'] else 'saved:') + metadata['owner']
            saved[key] += storage.nbytes()
            counts[key] += 1
        owned, seen = defaultdict(int), set()
        def visit(value, owner, depth=0):
            if torch.is_tensor(value):
                if is_fake(value) or value.layout != torch.strided:
                    return
                storage = value.untyped_storage()
                if storage._cdata not in seen:
                    seen.add(storage._cdata)
                    owned[owner] += storage.nbytes()
            elif depth < 3:
                if isinstance(value, dict):
                    for child in value.values(): visit(child, owner, depth + 1)
                elif isinstance(value, (tuple, list)):
                    for child in value: visit(child, owner, depth + 1)
        for name, value in model.named_parameters(): visit(value, 'parameters')
        for name, module in model.named_modules():
            for key, value in vars(module).items():
                if key not in ('_parameters', '_modules', '_model'):
                    visit(value, (name or 'model') + '.' + key)
        optimizer = getattr(model, '_sentence_optimizer', None)
        if optimizer is not None:
            for state in optimizer.state.values():
                for key, value in state.items(): visit(value, 'optimizer.' + key)
        record = dict(sequence=sequence, phase=phase, sentence=sid,
            physical_bytes=process.sample(), saved_total=sum(saved.values()),
            saved_owners=sorted(saved.items(), key=lambda item: -item[1]),
            saved_objects=sum(counts.values()), owned_total=sum(owned.values()),
            tensor_owners=sorted(owned.items(), key=lambda item: -item[1]))
        with destination.open('a') as stream: stream.write(json.dumps(record) + '\n')

    run = Models.BasicModel._run_batch_once
    @wraps(run)
    def batch(self, *args, **kwargs):
        identify(self)
        snapshot(self, 'batch-start')
        try:
            return run(self, *args, **kwargs)
        finally:
            snapshot(self, 'batch-end')
    commit = Models.BasicModel._commit_sentence
    def closing(self, state, sid, *args, **kwargs):
        snapshot(self, 'before-closing', sid)
        result = commit(self, state, sid, *args, **kwargs)
        snapshot(self, 'after-closing', sid)
        return result
    training_step = Models.BasicModel._sentence_train_step
    @wraps(training_step)
    def step(self, loss):
        snapshot(self, 'before-backward')
        result = training_step(self, loss)
        snapshot(self, 'after-backward')
        return result
    monkeypatch.setattr(Models.BasicModel, '_sentence_train_step', step)
    monkeypatch.setattr(SentenceCompose._SavedValue, '__init__', observe_saved)
    monkeypatch.setattr(Models.BasicModel, '_run_batch_once', batch)
    monkeypatch.setattr(Models.BasicModel, '_commit_sentence', closing)
    original.test_two_epoch_training_severs_cross_batch_graph(monkeypatch)
