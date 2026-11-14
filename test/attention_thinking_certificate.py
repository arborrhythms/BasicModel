"""Recorded-stream, exact certificates against 3926005f12; CPU only.

Run this file with the source checkout's bin and test first on PYTHONPATH.
The observer neither selects an initialization nor alters a fixture or bar.
Each output directory is immutable evidence, and --replay installs its stream.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch


def frozen(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().contiguous()
        # A singleton slice can be contiguous while retaining stride > 1.
        # Copy into an owned flat buffer before reinterpreting its bytes.
        raw = torch.empty(value.numel(), dtype=value.dtype)
        raw.copy_(value.reshape(-1))
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(raw.view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(key): frozen(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [frozen(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(type(value))


def parameters(model):
    return {name: frozen(value) for name, value in model.named_parameters()}


def stream():
    from random_streams import capture
    return capture()


def restore(state):
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])


def stream_digest():
    state = stream()
    return dict(torch=frozen(state['torch']), python=state['python'],
                numpy=[state['numpy'][0], state['numpy'][1].tolist(), *state['numpy'][2:]])


class Observer:
    def __init__(self, patch, output, *, certify):
        import Attention, ModelAttention, Language
        from Layers import Error
        from test_attention_thinking_steps import original_attention, exact
        self.output, self.certify = output, certify
        self.calls, self.actions, self.costs, self.records = [], [], [], []
        self.in_attention = 0
        self.reference_seconds = 0.
        base = original_attention()
        read = Attention.narrow_words
        def observed(chooser, keys, spans, known, **kwargs):
            initial = torch.get_rng_state()
            value = read(chooser, keys, spans, known, **kwargs)
            final = torch.get_rng_state()
            if certify:
                started = time.perf_counter()
                torch.set_rng_state(initial)
                wanted = base.narrow_words(chooser, keys, spans, known, **kwargs)
                exact(value, wanted)
                exact(torch.get_rng_state(), final)
                self.reference_seconds += time.perf_counter() - started
            self.calls.append(dict(inputs=frozen((keys, spans, known, kwargs)),
                result=frozen(value), actions=value.actions.tolist(),
                known=known.tolist(), spans=spans.tolist(),
                budget=kwargs['budget'], spent=value.table.spent.tolist(),
                departure=None if kwargs.get('departure') is None else kwargs['departure'].tolist()))
            return value
        patch.setattr(Attention, 'narrow_words', observed)
        patch.setattr(ModelAttention, 'narrow_words', observed)
        attend = Language.OperationSelectionLayer.attend
        def attending(chooser, *args, **kwargs):
            self.in_attention += 1
            try:
                return attend(chooser, *args, **kwargs)
            finally:
                self.in_attention -= 1
        patch.setattr(Language.OperationSelectionLayer, 'attend', attending)
        select = Language.OperationSelectionLayer.select_logits
        def selecting(chooser, *args, **kwargs):
            result = select(chooser, *args, **kwargs)
            if not self.in_attention:
                self.actions.append(frozen(result))
            return result
        patch.setattr(Language.OperationSelectionLayer, 'select_logits', selecting)
        for name in ('add', 'error'):
            original = getattr(Error, name)
            def register(registry, *args, _original=original, _name=name, **kwargs):
                result = _original(registry, *args, **kwargs)
                self.costs.append(dict(method=_name, args=frozen(args), kwargs=frozen(kwargs)))
                return result
            patch.setattr(Error, name, register)

    def batch(self, model, *, seconds, label, optimizer=None):
        index = len(self.records)
        record = dict(label=label, parameters=parameters(model), calls=self.calls,
                      actions=self.actions, costs=self.costs, stream=stream_digest())
        if optimizer is not None:
            record['optimizer'] = frozen(optimizer.state_dict())
        (self.output/f'batch-{index:03}.json').write_text(json.dumps(record) + '\n')
        self.records.append(dict(batch=index, label=label, seconds=seconds,
            attention_calls=len(self.calls), departure_calls=sum(c['departure'] is not None for c in self.calls),
            parameter_count=len(record['parameters']), costs=len(self.costs)))
        self.calls, self.actions, self.costs = [], [], []
        print(json.dumps(self.records[-1]), flush=True)


def lesson(observer):
    import Language
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    model = BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]
    model._tensor_peer_while_eager = True
    optimizer = model.getOptimizer(lr=.001)
    curriculum = json.loads(Path('data/corpus-lessons.json').read_text())
    texts, targets = stage_documents(TheData, curriculum['stages'][0])
    admit_training_forms(model, texts, targets)
    observer.calls, observer.actions, observer.costs = [], [], []
    run = model.runBatch
    seconds = 0.
    def timed(*args, **kwargs):
        nonlocal seconds
        overhead = observer.reference_seconds
        started = time.perf_counter()
        result = run(*args, **kwargs)
        seconds = time.perf_counter()-started-(observer.reference_seconds-overhead)
        return result
    model.runBatch = timed
    class Complete(Exception):
        pass
    def batch(model, split, rows, result):
        observer.batch(model, seconds=seconds, label=dict(rows=rows, texts=[texts[i] for i in rows]),
                       optimizer=optimizer)
        if len(observer.records) == 16:
            raise Complete
    try:
        # The full configured dataset is retained. This is the requested
        # first-sixteen observation boundary, not a smaller training fixture.
        present(model, split='train', optimizer=optimizer, batch_size=1, after_batch=batch)
    except Complete:
        pass
    finally:
        model.End()
    assert len(observer.records) == 16


def gates(observer, patch):
    import test_mm_xor
    import test_orders_learned
    import test_item25_completion
    models = []
    fresh = test_mm_xor._fresh_model
    def construct(*args, **kwargs):
        result = fresh(*args, **kwargs)
        models.append(result[0])
        return result
    patch.setattr(test_mm_xor, '_fresh_model', construct)
    step = torch.optim.Adam.step
    def update(optimizer, *args, **kwargs):
        started = time.perf_counter()
        result = step(optimizer, *args, **kwargs)
        observer.batch(models[-1], seconds=time.perf_counter()-started,
                       label='numeric XOR optimizer update', optimizer=optimizer)
        return result
    patch.setattr(torch.optim.Adam, 'step', update)
    test_mm_xor.TestMMXorConvergence.setUpClass()
    test_mm_xor.TestMMXorConvergence('test_convergence').test_convergence()
    owners = []
    fixture = test_orders_learned.equality_fixture
    def equality():
        result = fixture()
        owners.append(result[0])
        return result
    patch.setattr(test_orders_learned, 'equality_fixture', equality)
    started = time.perf_counter()
    test_item25_completion.test_a_previously_named_shared_concept_denotes_its_raised_class()
    observer.batch(owners[-1], seconds=time.perf_counter()-started, label='named shared class gate')
    for model in models:
        model.End()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('lesson', 'gates'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--certify', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        observer = Observer(patch, args.output, certify=args.certify)
        if args.replay:
            restore(torch.load(args.replay/'initial-rng.pt', weights_only=False))
            metadata = json.loads((args.replay/'initial.json').read_text())
            assert os.environ.get('PYTHONHASHSEED') == metadata['pythonhashseed']
        torch.save(stream(), args.output/'initial-rng.pt')
        (args.output/'initial.json').write_text(json.dumps(dict(
            pythonhashseed=os.environ.get('PYTHONHASHSEED'), seed_selected=False))+'\n')
        started = time.perf_counter()
        if args.kind == 'lesson':
            lesson(observer)
        else:
            gates(observer, patch)
        report = dict(kind=args.kind, pythonhashseed=os.environ.get('PYTHONHASHSEED'),
                      seed_selected=False, seconds=time.perf_counter()-started,
                      certificate_reference_seconds=observer.reference_seconds, records=observer.records)
        torch.save(stream(), args.output/'final-rng.pt')
        if args.compare:
            old = json.loads((args.compare/'result.json').read_text())
            assert len(old['records']) == len(observer.records)
            for i in range(len(observer.records)):
                name = f'batch-{i:03}.json'
                before = json.loads((args.compare/name).read_text())
                after = json.loads((args.output/name).read_text())
                assert after == before, f'exact batch certificate differs: {name}'
            report['matches_base_exactly'] = True
        (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
