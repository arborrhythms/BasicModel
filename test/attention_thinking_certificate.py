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
    def __init__(self, patch, output, *, certify, require_word_driven=False):
        import Attention, ModelAttention, Language
        from Layers import Error
        from test_attention_thinking_steps import original_attention, exact
        self.output, self.certify = output, certify
        self.require_word_driven = require_word_driven
        self.calls, self.actions, self.costs, self.records = [], [], [], []
        self.walk_steps = []
        self.word_driven = []
        self.read_both = 0
        self.in_attention = 0
        self.reference_seconds = 0.
        base = original_attention() if certify else None
        opened = []
        if hasattr(Attention, 'open_narrowing'):
            opening = Attention.open_narrowing
            def observe_open(*args, **kwargs):
                state = opening(*args, **kwargs)
                state._certificate_arguments = (
                    state.chooser, state.keys, state.spans, state.known,
                    {key: value for key, value in kwargs.items()
                     if key not in ('chooser', 'keys', 'spans', 'known')})
                if certify:
                    opened.append(state)
                return state
            patch.setattr(Attention, 'open_narrowing', observe_open)
            if hasattr(ModelAttention, 'open_narrowing'):
                patch.setattr(ModelAttention, 'open_narrowing', observe_open)
        read = Attention.narrow_words
        def recorded(chooser, keys, spans, known, kwargs, value, initial, final, steps):
            self.walk_steps.append(steps)
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
        def observed(chooser, keys, spans, known, **kwargs):
            initial = torch.get_rng_state()
            value = read(chooser, keys, spans, known, **kwargs)
            final = torch.get_rng_state()
            steps = opened.pop().steps_taken if opened else kwargs['budget']
            return recorded(chooser, keys, spans, known, kwargs, value, initial, final, steps)
        patch.setattr(Attention, 'narrow_words', observed)
        patch.setattr(ModelAttention, 'narrow_words', observed)
        # A known word read both: its for and its against lane above zero
        # at the walk's read of the native word poles.
        poles_of = ModelAttention.native_word_poles
        def counted_poles(model, spans, forms, known):
            poles = poles_of(model, spans, forms, known)
            lanes = poles[known.to(poles.device)]
            self.read_both += int(((lanes[..., 0] > 0) & (lanes[..., 1] > 0)).sum())
            return poles
        patch.setattr(ModelAttention, 'native_word_poles', counted_poles)
        self.attend_count = 0
        from SentenceLooks import Looks
        import ConceptSpread
        expected = ConceptSpread.expected
        expect_calls = []
        def counted_expect(owner, knowing):
            expect_calls.append(len(knowing))
            return expected(owner, knowing)
        patch.setattr(ConceptSpread, 'expected', counted_expect)
        native_run = Looks.run
        stepping = Attention.narrow_step
        seen_states = []
        def table_state(state):
            table = state.table
            return hashlib.sha256(b''.join(
                value.detach().cpu().contiguous().numpy().tobytes() for value in (
                    table.intervals, table.valid, table.space, table.level, table.done,
                    state.accepted, state.descended, state.reductions, state.field_valid))).hexdigest()
        def stepped(state, **kwargs):
            if seen_states:
                seen_states[-1].append(table_state(state))
            return stepping(state, **kwargs)
        patch.setattr(Attention, 'narrow_step', stepped)
        def measured_path(path, arrive, **kwargs):
            if certify:
                raise ValueError('the historical acceptance oracle cannot certify the loop of looks')
            state = path.walk.state
            words = []
            first_expect = len(expect_calls)
            before = self.attend_count
            def arrived(index, rows):
                words.append(dict(index=int(index), rows=rows.tolist(), after_attend=self.attend_count-before))
                return arrive(index, rows)
            seen_states.append([])
            try:
                result = native_run(path, arrived, **kwargs)
            finally:
                states = seen_states.pop()
            steps = self.attend_count-before
            self.walk_steps.append(steps)
            self.calls.append(dict(actions=result.actions.tolist(), spent=result.table.spent.tolist(),
                                   departure=None if path.draw is None else frozen(path.draw)))
            self.word_driven.append(dict(looks=len(path.fields), attend_calls=steps, words=words,
                repeated_states=len(states) - len(set(states)),
                look_expect_calls=len(expect_calls) - first_expect,
                look_expect_forecasts=sum(expect_calls[first_expect:])))
            return result
        patch.setattr(Looks, 'run', measured_path)
        attend = Language.OperationSelectionLayer.attend
        def attending(chooser, *args, **kwargs):
            self.attend_count += 1
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
                term = args[0] if args else kwargs['name']
                self.costs.append(dict(method=_name, args=frozen(args), kwargs=frozen(kwargs),
                    registered=frozen(registry._terms.get(term)), rows=frozen(registry.row_mask)))
                return result
            patch.setattr(Error, name, register)

    def batch(self, model, *, seconds, label, optimizer=None):
        if self.require_word_driven and isinstance(label, dict):
            assert self.word_driven, 'sentence attention was completed outside the word loop'
        index = len(self.records)
        record = dict(label=label, parameters=parameters(model), calls=self.calls,
                      actions=self.actions, costs=self.costs, stream=stream_digest())
        if optimizer is not None:
            record['optimizer'] = frozen(optimizer.state_dict())
        (self.output/f'batch-{index:03}.json').write_text(json.dumps(record) + '\n')
        self.records.append(dict(batch=index, label=label, seconds=seconds,
            attention_calls=len(self.calls), departure_calls=sum(c['departure'] is not None for c in self.calls),
            parameter_count=len(record['parameters']), costs=len(self.costs), walk_steps=self.walk_steps,
            word_driven=self.word_driven, known_read_both=self.read_both))
        self.calls, self.actions, self.costs = [], [], []
        self.walk_steps = []
        self.word_driven = []
        self.read_both = 0
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


def over_runs(output, runs):
    """The lesson run ``runs`` times, each in a fresh process with no seed
    chosen, into ``output/run-NN``, and its report over the runs: the words
    found in the first walk of each of the sixteen sentences, summed; the
    first walks that ran to the cap of 32 steps; the first walks that came
    back to a state of their table they had been in; the readings of a
    known word whose for and against lanes were both above zero; and the
    seconds."""
    import subprocess
    import sys
    output.mkdir(parents=True, exist_ok=False)
    each = []
    for k in range(runs):
        where = output/f'run-{k + 1:02}'
        subprocess.run([sys.executable, __file__, 'lesson', '--output', str(where)],
                       check=True, stdout=subprocess.DEVNULL)
        result = json.loads((where/'result.json').read_text())
        first = [record['word_driven'][0] if record['word_driven'] else None
                 for record in result['records']]
        steps = [record['walk_steps'][0] if record['walk_steps'] else None
                 for record in result['records']]
        each.append(dict(run=k + 1, seconds=result['seconds'],
                         words=sum(len(walk['words']) for walk in first if walk),
                         at_cap=sum(step == 32 for step in steps),
                         repeating=sum(bool(walk.get('repeated_states')) for walk in first if walk),
                         known_read_both=sum(record.get('known_read_both', 0) for record in result['records']),
                         first_walk_steps=steps))
        print(json.dumps(each[-1]), flush=True)
    words = [run['words'] for run in each]
    capped = [run['at_cap'] for run in each]
    report = dict(runs=runs, seed_selected=False, each=each,
                  words=dict(each=words, mean=sum(words) / runs, least=min(words), greatest=max(words)),
                  at_cap=dict(each=capped, mean=sum(capped) / runs, total=sum(capped)),
                  repeating=dict(each=[run['repeating'] for run in each],
                                 total=sum(run['repeating'] for run in each)),
                  known_read_both=dict(each=[run['known_read_both'] for run in each],
                                       total=sum(run['known_read_both'] for run in each)),
                  seconds=dict(mean=sum(run['seconds'] for run in each) / runs))
    (output/'runs.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'each'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('lesson', 'gates'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--certify', action='store_true')
    parser.add_argument('--require-word-driven', action='store_true')
    parser.add_argument('--runs', type=int, default=1,
                        help='run the lesson this many times, each in a fresh process, and report over the runs')
    args = parser.parse_args()
    if args.runs > 1:
        if args.kind != 'lesson' or args.replay or args.compare or args.certify:
            parser.error('--runs reports over fresh lessons alone')
        over_runs(args.output, args.runs)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        observer = Observer(patch, args.output, certify=args.certify,
                            require_word_driven=args.require_word_driven)
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
