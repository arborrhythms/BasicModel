"""The reading lesson's loop of requests, recorded batch by batch; CPU only.

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


def draw_start():
    """A start for the random streams, drawn from the system's entropy."""
    start = int.from_bytes(os.urandom(8), 'little') % (2 ** 32)
    torch.manual_seed(start)
    np.random.seed(start)
    random.seed(start)
    return start


def stream_digest():
    state = stream()
    return dict(torch=frozen(state['torch']), python=state['python'],
                numpy=[state['numpy'][0], state['numpy'][1].tolist(), *state['numpy'][2:]])


class Observer:
    """Records each sentence's loop of requests: the looks each path took,
    the words it read and in what order, the looks that drew among equals,
    and the lanes of each known word at the first look that held it."""
    def __init__(self, patch, output, *, require_word_driven=False):
        import Language
        import SentenceLooks
        from Layers import Error
        from SentenceLooks import Looks
        self.output = output
        self.require_word_driven = require_word_driven
        self.calls, self.actions, self.costs, self.records = [], [], [], []
        self.walk_steps = []
        self.word_driven = []
        self.in_attention = 0
        self.attend_count = 0
        lanes_seen = []
        activity = SentenceLooks.word_activity
        def observed_activity(field, concepts, known, extent, own, bank):
            if lanes_seen:
                B, W = concepts.shape
                ids = field.concept_ids
                ids = ids[:, None].expand(-1, B) if ids.ndim == 1 else ids
                evidence = field.evidence[:, :, :W].detach()                     # [S, B, W, 2]
                match = (ids[:, :, None] == concepts[None]) & (concepts[None] >= 0)
                lanes = torch.where(match[..., None], evidence, 0.).amax(0)       # [B, W, 2]
                width = min(W, extent.shape[1])
                for b, w in (known[:, :width] & extent[:, :width]).nonzero().tolist():
                    lanes_seen[-1].setdefault((b, w), lanes[b, w].tolist())
            return activity(field, concepts, known, extent, own, bank)
        patch.setattr(SentenceLooks, 'word_activity', observed_activity)
        native_run = Looks.run
        def measured_path(path, arrive, **kwargs):
            state = path.walk
            before = self.attend_count
            arrivals = []
            def arrived(index, rows):
                arrivals.append(dict(index=int(index), rows=rows.tolist()))
                return arrive(index, rows)
            lanes_seen.append({})
            try:
                result = native_run(path, arrived, **kwargs)
            finally:
                lanes = lanes_seen.pop()
            live = ((state.spans[..., 1] > state.spans[..., 0]) & path.present[:, None])
            orders = path.source_orders()
            # The arrivals against the live source slots (sigma-pi §19.124, 3).
            from AttentionLesson import in_order
            ordered = in_order(orders, state.spans)
            streams = []
            for b in range(len(path.present)):
                if not bool(path.present[b]):
                    continue
                order = list(orders[b])
                count = int(live[b].sum())
                known = [(w, lane) for (row, w), lane in sorted(lanes.items()) if row == b]
                streams.append(dict(
                    words=len(order), live=count, order=order,
                    in_order=bool(ordered[b]),
                    looks=int(result.looks[b]), allowance=int(state.limit[b]),
                    at_allowance=bool(int(state.taken) >= int(state.limit[b])),
                    drawn=int(result.alternatives[b].sum()),
                    known_against=sum(1 for _, (f, a) in known if a > f),
                    known_both=sum(1 for _, (f, a) in known if f > 0 and a > 0)))
            self.walk_steps.append(self.attend_count - before)
            self.calls.append(dict(actions=result.actions.tolist(), looks=result.looks.tolist(),
                                   departure=None if path.draw is None else frozen(path.draw)))
            self.word_driven.append(dict(looks=len(path.fields), attend_calls=self.attend_count - before,
                                         words=arrivals, streams=streams))
            return result
        patch.setattr(Looks, 'run', measured_path)
        attend = Language.OperationSelectionLayer.attend
        def attending(*args, **kwargs):
            self.attend_count += 1
            self.in_attention += 1
            try:
                return attend(*args, **kwargs)
            finally:
                self.in_attention -= 1
        patch.setattr(Language.OperationSelectionLayer, 'attend', staticmethod(attending))
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
            word_driven=self.word_driven))
        self.calls, self.actions, self.costs = [], [], []
        self.walk_steps = []
        self.word_driven = []
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
        started = time.perf_counter()
        result = run(*args, **kwargs)
        seconds = time.perf_counter()-started
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


def summary(records):
    """Over the first path of each sentence: the words found, the sentences
    and words read in order, the first paths that took the allowance of
    looks, the looks that drew among equals, and the known words whose
    first look held them read against (against above for) or both (both
    lanes above zero)."""
    first = [stream for record in records if record['word_driven']
             for stream in record['word_driven'][0]['streams']]
    return dict(sentences=len(first),
                words=sum(s['words'] for s in first),
                live=sum(s['live'] for s in first),
                in_order=sum(s['in_order'] for s in first),
                words_in_order=sum(s['words'] for s in first if s['in_order']),
                at_allowance=sum(s['at_allowance'] for s in first),
                drawn=sum(s['drawn'] for s in first),
                known_against=sum(s['known_against'] for s in first),
                known_both=sum(s['known_both'] for s in first))


def over_runs(output, runs):
    """The lesson run ``runs`` times, each in a fresh process with no seed
    chosen and its start drawn from the system's entropy, into
    ``output/run-NN``, and its report over the runs: the start each drew,
    the summary of each run's sixteen sentences, and the seconds."""
    import subprocess
    import sys
    output.mkdir(parents=True, exist_ok=False)
    each = []
    for k in range(runs):
        where = output/f'run-{k + 1:02}'
        subprocess.run([sys.executable, __file__, 'lesson', '--output', str(where)],
                       check=True, stdout=subprocess.DEVNULL)
        result = json.loads((where/'result.json').read_text())
        start = json.loads((where/'initial.json').read_text()).get('start_drawn')
        each.append(dict(run=k + 1, start_drawn=start, seconds=result['seconds'], **summary(result['records'])))
        print(json.dumps(each[-1]), flush=True)
    keys = [key for key in each[0] if key not in ('run', 'seconds', 'start_drawn')]
    report = dict(runs=runs, seed_selected=False, each=each,
                  **{key: dict(each=[run[key] for run in each], total=sum(run[key] for run in each),
                               least=min(run[key] for run in each), greatest=max(run[key] for run in each))
                     for key in keys},
                  seconds=dict(mean=sum(run['seconds'] for run in each) / runs))
    (output/'runs.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'each'}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('lesson', 'gates'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--require-word-driven', action='store_true')
    parser.add_argument('--runs', type=int, default=1,
                        help='run the lesson this many times, each in a fresh process, and report over the runs')
    args = parser.parse_args()
    if args.runs > 1:
        if args.kind != 'lesson' or args.replay or args.compare:
            parser.error('--runs reports over fresh lessons alone')
        over_runs(args.output, args.runs)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        observer = Observer(patch, args.output,
                            require_word_driven=args.require_word_driven)
        start = None
        if args.replay:
            restore(torch.load(args.replay/'initial-rng.pt', weights_only=False))
            metadata = json.loads((args.replay/'initial.json').read_text())
            assert os.environ.get('PYTHONHASHSEED') == metadata['pythonhashseed']
        else:
            # A fresh process starts the same random streams; each run draws
            # its start from the system's entropy, and says what it drew.
            start = draw_start()
            print(json.dumps(dict(start_drawn=start)), flush=True)
        torch.save(stream(), args.output/'initial-rng.pt')
        (args.output/'initial.json').write_text(json.dumps(dict(
            pythonhashseed=os.environ.get('PYTHONHASHSEED'), seed_selected=False, start_drawn=start))+'\n')
        started = time.perf_counter()
        if args.kind == 'lesson':
            lesson(observer)
        else:
            gates(observer, patch)
        report = dict(kind=args.kind, pythonhashseed=os.environ.get('PYTHONHASHSEED'),
                      seed_selected=False, seconds=time.perf_counter()-started,
                      summary=summary(observer.records), records=observer.records)
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
