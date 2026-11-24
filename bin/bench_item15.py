"""Full reading updates, with timing and a separately profiled update.

Run from the checkout with BASICMODEL_DEVICE=cpu MODEL_COMPILE=none
PYTHONHASHSEED=0. A fixed seed is for paired measurements, never a test gate.
"""
import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=1505)
    parser.add_argument('--explore-paths', type=int, help='override the candidate count in a benchmark-only config')
    parser.add_argument('--batch-size', type=int, help='study a different batch width using distinct lesson pages')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--lesson', action='store_true')
    parser.add_argument('--training-seconds', type=float,
                        help='assess a fresh lesson reader after this budget of complete training updates')
    parser.add_argument('--corpus', choices=('lesson-13', 'math-chain'))
    parser.add_argument('--path-audit', action='store_true', help='record candidate-table changes during individual trial walks')
    parser.add_argument('--dispatch-census', action='store_true',
                        help='count native host-visible results, variable-size outputs and copies on another update')
    parser.add_argument('--allocations', action='store_true', help='profile allocation operators on a separate CPU step')
    parser.add_argument('--compiled-step', action='store_true',
                        help='enable the native compiled entry (step mode only)')
    parser.add_argument('--strict-sentence', choices=('eager', 'inductor'),
                        help='probe fullgraph capture of the entire normal input sentence on one training update')
    parser.add_argument('--compile-percept-fold', action='store_true',
                        help='also compile the live whole/part fold for a native step comparison')
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-revision', help='revision of an archived --source without Git metadata')
    args = parser.parse_args()
    if args.training_seconds is not None and (not args.lesson or args.training_seconds <= 0):
        parser.error('--training-seconds needs --lesson and a positive training time')
    if args.dispatch_census and (args.lesson or args.corpus):
        parser.error('--dispatch-census requires step mode')
    if args.strict_sentence and (args.lesson or args.corpus or args.compiled_step):
        parser.error('--strict-sentence probes the ordinary unpartitioned step')
    if args.lesson and args.compiled_step:
        parser.error('--lesson uses the native lesson runner; --compiled-step requires step mode')
    if args.compile_percept_fold and not args.compiled_step:
        parser.error('--compile-percept-fold requires --compiled-step')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    sys.path.insert(0, str(args.source.resolve() / 'bin'))
    from pretrain_reading import ROOT, LESSON, load_lesson, run
    from Models import BaseModel
    from data import Data
    from What import What
    spec = load_lesson(LESSON)
    config = ROOT / 'data/XOR_grammar.xml'
    lesson = LESSON
    if args.lesson and args.batch_size is not None:
        if args.batch_size < 1:
            parser.error('--batch-size must be positive')
        spec['batch_size'] = args.batch_size
        lesson = args.output.resolve() / 'lesson.json'
        lesson.write_text(json.dumps(spec, indent=2) + '\n')
    if args.corpus:
        if args.lesson or args.batch_size or args.explore_paths is not None or args.compiled_step:
            parser.error('--corpus uses its own lesson configuration')
        config = ROOT / 'data' / ('MM_corpus_lessons.xml' if args.corpus == 'lesson-13' else 'MM_math_chain.xml')
    if args.explore_paths is not None:
        import xml.etree.ElementTree as ET
        if args.explore_paths < 0:
            parser.error('--explore-paths must be nonnegative')
        tree = ET.parse(config)
        architecture = tree.getroot().find('architecture')
        setting = architecture.find('explorePaths')
        if setting is None:
            setting = ET.SubElement(architecture, 'explorePaths')
        setting.text = str(args.explore_paths)
        config = args.output.resolve() / 'benchmark.xml'
        tree.write(config, encoding='utf-8', xml_declaration=True)
    record = dict(source=args.source_revision or subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                  seed=args.seed, torch=torch.__version__, platform=platform.platform(),
                  threads=torch.get_num_threads(), batch_size=spec['batch_size'],
                  compiled_step=args.compiled_step, strict_sentence=args.strict_sentence,
                  config=str(config), config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
                  lesson_sha256=hashlib.sha256(lesson.read_bytes()).hexdigest(),
                  environment={k: v for k, v in os.environ.items() if k.startswith(('MODEL_', 'BASICMODEL_', 'PYTHONHASH', 'TORCHINDUCTOR'))},
                  steps=[])
    record['explore_paths_override'] = args.explore_paths
    record['source_files'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((ROOT / 'bin').glob('*.py'))}

    def save():
        (args.output / 'timing.json').write_text(json.dumps(record, indent=2) + '\n')

    # runBatch belongs to BasicModel; observe the actual entry, not a kernel.
    from Models import BasicModel
    if args.compile_percept_fold:
        import BasicPercepts
        from CompilerEffects import compile_inductor, retained_backward_compiler
        BasicPercepts.held_at = torch.compile(BasicPercepts.held_at,
            backend=retained_backward_compiler(compile_inductor), fullgraph=True, dynamic=True)
        record['compile_percept_fold'] = True
    original = BasicModel.runBatch

    class TrainingBudgetReached(Exception):
        pass

    train_budget_active = False
    training_time = 0.

    def sync(device):
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize(device)

    def measured(model, *a, **kw):
        nonlocal training_time
        device = next(model.parameters()).device
        record['device'] = str(device)
        from torch._dynamo.utils import compilation_time_metrics, counters
        before = dict(counters['stats'])
        # These are the outer frame compiler and the separately timed lazy
        # backward compiler. Nested compiler passes overlap those intervals.
        timers = ('_compile.compile_inner', 'backward._backward_impl')
        offsets = [len(compilation_time_metrics.get(name, ())) for name in timers]
        updates = getattr(model, '_training_step_count', 0)
        sync(device)
        start = time.perf_counter()
        result = original(model, *a, **kw)
        sync(device)
        seconds = time.perf_counter() - start
        training = kw.get('train', False)
        # Count the actual occupied sentence slots after the timed update.
        # A packed page can close several sentences. Padding closes none.
        slots = model.inputSpace._packed_sentence_slot_mask
        row = dict(seconds=seconds,
                   updates=getattr(model, '_training_step_count', 0) - updates,
                   graphs=counters['stats'].get('unique_graphs', 0) - before.get('unique_graphs', 0),
                   train=training, batch_size=kw.get('batchSize'),
                   sentences=int(slots.sum()) if training else 0)
        row['compiler_seconds'] = sum(sum(compilation_time_metrics.get(name, ())[offset:])
                                      for name, offset in zip(timers, offsets, strict=True))
        record['steps'].append(row)
        save()
        print('step', len(record['steps']), row, flush=True)
        if train_budget_active and row['train']:
            training_time += row['seconds']
            if training_time >= args.training_seconds:
                raise TrainingBudgetReached
        return result

    BasicModel.runBatch = measured
    start = time.perf_counter()
    if args.corpus:
        from dataclasses import asdict
        from corpus_lessons import measure, stage_documents
        from math_chain_corpus import MathChainCorpus
        from MathChainTraining import present
        from data import TheData
        record.update(workload=args.corpus, batch_size=1)
        if args.corpus == 'lesson-13':
            corpus = ROOT / 'data/corpus-lessons.json'
            record['lesson_sha256'] = hashlib.sha256(corpus.read_bytes()).hexdigest()
            measure(config, corpus, args.output / 'curriculum', eager=True, presentations=1)
        else:
            corpus = MathChainCorpus()
            corpus.generator.rng.seed(args.seed)
            document = corpus.problem((3, 5), split='train', training=True)
            record['document'] = asdict(document)
            record.pop('lesson_sha256')
            data = TheData
            data.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                            for split in ('train', 'validation', 'test')})
            model, _ = BaseModel.from_config(str(config), data=data)
            try:
                stage_documents(data, {'documents': [{'document': document.key, 'records': [asdict(document)]}]})
                optimizer = model.getOptimizer()
                model._optimizer = optimizer
                record['presentation'] = present(model, split='train', optimizer=optimizer, batch_size=1)
            finally:
                model.End()
    elif args.lesson:
        import pretrain_reading
        train_reader = pretrain_reading.train_reader
        def budgeted_reader(model, spec, optimizer):
            nonlocal train_budget_active
            train_budget_active = True
            try:
                return train_reader(model, spec, optimizer)
            except TrainingBudgetReached:
                record['training_budget_reached'] = True
                return [pretrain_reading.measure(model, spec['validation'], spec['batch_size'])]
            finally:
                train_budget_active = False
        if args.training_seconds is not None:
            pretrain_reading.train_reader = budgeted_reader
        try:
            report = run(config, lesson, args.output / 'reader')
        finally:
            pretrain_reading.train_reader = train_reader
        record['training_budget_seconds'] = args.training_seconds
        record['training_budget_used'] = training_time if args.training_seconds is not None else None
        record['lesson_seconds'] = time.perf_counter() - start
        record['passed'] = report['passed']
    else:
        model, _ = BaseModel.from_config(str(config), data=Data())
        if args.compiled_step:
            model.enable_compiled_step()
        record['construction_seconds'] = time.perf_counter() - start
        record['device'] = str(next(model.parameters()).device)
        optimizer = model.getOptimizer()
        model._optimizer = optimizer
        model.train()
        texts = ['hello world hello', 'world hello world', 'red blue red', 'blue red blue']
        if args.batch_size is not None:
            pages = list(dict.fromkeys([*texts, *spec['train']]))
            if not 1 <= args.batch_size <= len(pages):
                parser.error('batch size must fit the available distinct pages')
            texts = pages[:args.batch_size]
        record['batch_size'] = len(texts)
        record['explore_paths'] = model.explore_paths
        record['texts'] = texts
        data = model.inputSpace.data

        def step(*, time_boundaries=False):
            supervised = data.has_supervised_outputs
            if time_boundaries:
                step_index = len(record['steps'])
                device = next(model.parameters()).device
                sync(device)
                setup_start = time.perf_counter()
            try:
                with data.runtime_batch(texts):
                    data.has_supervised_outputs = False
                    source = model.inputSpace.prepInput(texts)
                    record['input_shape'] = list(source.shape)
                    questions = tuple(What.present(i, split='runtime') for i in range(len(texts)))
                    batch = source, torch.empty(len(texts), 0, device=source.device)
                    if time_boundaries:
                        sync(device)
                        setup_seconds = time.perf_counter() - setup_start
                    return model.runBatch(train=True, optimizer=optimizer, batchSize=len(texts),
                        split='runtime', questions=questions, batch_override=batch)
            finally:
                if time_boundaries:
                    reset_start = time.perf_counter()
                data.has_supervised_outputs = supervised
                model.End()
                model.dispatch_per_row_reset([True] * len(texts))
                if time_boundaries and len(record['steps']) > step_index:
                    sync(device)
                    record['steps'][-1].update(input_setup_seconds=setup_seconds,
                                               end_and_reset_seconds=time.perf_counter() - reset_start)

        if args.strict_sentence:
            method = torch.compile(BasicModel._execute_input_sentence, backend=args.strict_sentence,
                                   fullgraph=True)
            try:
                with patch.object(BasicModel, '_execute_input_sentence', method):
                    step()
            except Exception as error:
                record['capture_error'] = str(error)
                raise
            finally:
                save()
            return

        def fixed_storage():
            owner = model._concept_owner()
            pi, sigma = owner.basic_percepts.pi, owner.order_zero
            arrays = [*(('pi.' + n, getattr(pi, n)) for n in ('col', 'crow', 'value', 'against')),
                      *(('sigma.' + n, getattr(sigma, n)) for n in ('entry_input', 'entry_row'))]
            return {name: (tuple(t.shape), t.untyped_storage()._cdata) for name, t in arrays}

        try:
            storage = fixed_storage()
            record['fixed_store_resizes'] = []
            for update in range(args.steps):
                step(time_boundaries=True)
                current = fixed_storage()
                record['fixed_store_resizes'].extend((update, n) for n in current if current[n] != storage[n])
                storage = current
            buffers = {n: p.detach().cpu().clone() for n, p in model.named_buffers()}
            # Older readers stored this mask as a set of observed bytes.
            # Record its actual contents so allocation changes remain comparable.
            for name, module in model.named_modules():
                if type(module).__name__ == 'WordIdentity' and not hasattr(module, '_witness_counts'):
                    from WordIdentity import text_of
                    counts = torch.full((module.store._capacity,), -1, dtype=torch.long, device='cpu')
                    for form, row in module.word_rows.items():
                        if text_of(form).replace(' ', '').isalpha():
                            counts[row] = module.word_witnesses.get(form, 0)
                    buffers[f'{name}._witness_counts'] = counts
                if type(module).__name__ == 'WholeSpace' and not hasattr(module, '_standalone_bytes_seen'):
                    seen = torch.zeros(256, dtype=torch.bool, device='cpu')
                    seen[list(getattr(module, '_standalone_run_bytes', ()))] = True
                    buffers[f'{name}._standalone_bytes_seen'] = seen
            torch.save(dict(parameters={n: p.detach().cpu().clone() for n, p in model.named_parameters()},
                            buffers=buffers,
                            initialized_buffer_lengths={f'{n}.{suffix}': sum(module._priming_bridge_counts)
                                for n, module in model.named_modules() if hasattr(module, '_priming_bridge_counts')
                                for suffix in ('_priming_bridge_src', '_priming_bridge_dst', '_priming_bridge_bits')},
                            optimizer=optimizer.state_dict(),
                            rng=torch.get_rng_state()), args.output / 'updated.pt')
            # Instrumentation is outside the timing sample, on the next update.
            BasicModel.runBatch = original
            profiler = cProfile.Profile()
            profiler.runcall(step)
            profiler.dump_stats(str(args.output / 'update.prof'))
            stats = pstats.Stats(profiler)
            rows = []
            for (file, line, name), (primitive, calls, own, cumulative, _) in stats.stats.items():
                rows.append(dict(file=file, line=line, name=name, calls=calls,
                                 own_seconds=own, cumulative_seconds=cumulative))
            record['profile'] = sorted(rows, key=lambda r: -r['cumulative_seconds'])
            record['host_reads'] = [r for r in rows if any(token in r['name'] for token in ("'tolist'", "'item'", "'cpu'", "'__bool__'", "'__int__'", "'__float__'"))]
            # A second, untimed update counts implicit Python scalar reads
            # that cProfile's C-call events do not report. Counter bookkeeping
            # never reads a tensor to identify a repeated lookup.
            import SentenceLooks
            from Spaces import Codebook
            host, at_look, calls = Counter(), Counter(), Counter()
            in_look = [False]
            sentence_depth = 0
            in_sentence, sentence_sites = Counter(), Counter()
            execute_sentence = BasicModel._execute_input_sentence

            def sentence(*a, **kw):
                nonlocal sentence_depth
                sentence_depth += 1
                try:
                    return execute_sentence(*a, **kw)
                finally:
                    sentence_depth -= 1
            sites, percept_addresses = Counter(), Counter()
            sentence_callers = Counter()
            source_directory = str(args.source.resolve() / 'bin') + os.sep
            methods = {name: getattr(torch.Tensor, name) for name in
                       ('tolist', 'item', 'cpu', '__bool__', '__int__', '__float__')}
            def counted(name, function):
                def invoke(*a, **kw):
                    host[name] += 1
                    if sentence_depth:
                        caller = sys._getframe(1)
                        in_sentence[name] += 1
                        site = (Path(caller.f_code.co_filename).name, caller.f_code.co_name, str(caller.f_lineno), name)
                        sentence_sites[site] += 1
                        if args.dispatch_census and caller.f_code.co_filename.startswith(source_directory):
                            chain = []
                            frame = caller.f_back
                            while frame is not None and len(chain) < 3:
                                code = frame.f_code
                                if code.co_filename.startswith(source_directory):
                                    chain.append(f'{Path(code.co_filename).name}:{code.co_name}:{frame.f_lineno}')
                                frame = frame.f_back
                            sentence_callers[(':'.join(site), *chain)] += 1
                    if in_look[0]:
                        at_look[name] += 1
                        caller = sys._getframe(1)
                        sites[(Path(caller.f_code.co_filename).name, caller.f_code.co_name, name)] += 1
                    return function(*a, **kw)
                return invoke
            lookup, looks = Codebook.lookup_rows, SentenceLooks.Looks.run
            def rows_read(book, indices, **kw):
                caller = sys._getframe(1)
                site = (Path(caller.f_code.co_filename).name, caller.f_code.co_name)
                calls[site] += 1
                if site[0] == 'BasicPercepts.py' and site[1] in (
                        '_boundary_codes', '_stored_boundary_codes', '_occurrence_what'):
                    # The separate census is untimed. Bypass its host counter
                    # for this instrumentation-only read of the requested IDs.
                    ids = methods['tolist'](indices.detach().reshape(-1))
                    percept_addresses.update((id(book), row) for row in ids)
                return lookup(book, indices, **kw)
            def growth_state():
                pool = model._concept_owner().mtm
                return {name: dict(counts=matrix.counts(), entries={str(k): vars(v).copy() for k, v in
                                    matrix.entries.items()}, tallies={str(k): v.copy() for k, v in matrix.tallies.items()})
                        for name, matrix in (getattr(pool, 'candidates', None) or {}).items()}
            trial_audit = []
            def looking(walk, *a, **kw):
                in_look[0] = True
                try:
                    if args.path_audit:
                        previous = growth_state()
                        predictor = model.percept_expectation
                        counts = predictor.context_count[:len(walk.present)].detach().clone()
                        updates = predictor.updates.detach().clone()
                    result = looks(walk, *a, **kw)
                    if args.path_audit:
                        after = growth_state()
                        # These diagnostic reads use the original tensor
                        # method so they do not count as model host calls.
                        indices = [methods['tolist'](torch.where(step['interpreted'], step['index'], -1))
                                   for step in walk.decisions]
                        shared_arrivals = methods['tolist'](predictor.context_count[:len(counts)] - counts)
                        shared_updates = methods['tolist'](predictor.updates - updates)
                        trial_audit.append(dict(kind=model._sentence_trial, shared_tables_changed=previous != after,
                                                before=previous, after=after, interpreted_by_look=indices,
                                                shared_native_arrivals=shared_arrivals,
                                                shared_native_updates=shared_updates))
                    return result
                finally:
                    in_look[0] = False
            try:
                for name, function in methods.items():
                    setattr(torch.Tensor, name, counted(name, function))
                BasicModel._execute_input_sentence = sentence
                Codebook.lookup_rows, SentenceLooks.Looks.run = rows_read, looking
                step()
            finally:
                for name, function in methods.items():
                    setattr(torch.Tensor, name, function)
                BasicModel._execute_input_sentence = execute_sentence
                Codebook.lookup_rows, SentenceLooks.Looks.run = lookup, looks
            if args.path_audit:
                record['trial_growth'] = trial_audit
                record['selected_paths'] = model._last_sentence_credit['winner'].tolist()
            record['percept_addresses'] = dict(total=percept_addresses.total(),
                distinct=len(percept_addresses), repeated=percept_addresses.total() - len(percept_addresses))
            record['host_census'] = dict(update=dict(host), sentence=dict(in_sentence), looks=dict(at_look),
                                         sentence_reads_by_caller={':'.join(k): v for k, v in sentence_sites.most_common()},
                                         look_reads_by_caller={':'.join(k): v for k, v in sites.most_common()},
                                         lookups_by_caller={':'.join(k): v for k, v in calls.items()})
            if args.dispatch_census:
                record['host_census']['sentence_caller_chains'] = {
                    ' <- '.join(k): v for k, v in sentence_callers.most_common()}
            # Capture live gradients at every optimizer boundary in a separate
            # complete update. runBatch clears .grad before returning, so a
            # post-update .grad snapshot would certify nothing.
            optimizer_step, gradients = optimizer.step, []
            def audited_step(*a, **kw):
                gradients.append({n: None if p.grad is None else p.grad.detach().cpu().clone()
                                  for n, p in model.named_parameters()})
                return optimizer_step(*a, **kw)
            optimizer.step = audited_step
            try:
                step()
            finally:
                optimizer.step = optimizer_step
            record['gradient_audit'] = dict(steps=len(gradients),
                live_tensors=[sum(g is not None for g in snapshot.values()) for snapshot in gradients])
            torch.save(gradients, args.output / 'gradients.pt')
            if args.path_audit:
                # A separate complete update measures the real path callbacks
                # without the Python host-read census or cProfile installed.
                import WholePaths
                device = next(model.parameters()).device
                durations = []

                def timed(function, phase, branch, *a, **kw):
                    sync(device)
                    start = time.perf_counter()
                    result = function(*a, **kw)
                    sync(device)
                    durations.append(dict(phase=phase, branch=branch, seconds=time.perf_counter() - start))
                    return result

                finish = WholePaths.finish
                def timed_looks(walk, *a, **kw):
                    return timed(looks, 'looks', 'explore' if walk.prior is not None else 'exploit', walk, *a, **kw)

                def timed_score(*a, **kw):
                    return timed(finish, 'score', 'explore' if kw.get('base') is not None else 'exploit', *a, **kw)

                with patch.object(SentenceLooks.Looks, 'run', timed_looks), patch.object(WholePaths, 'finish', timed_score):
                    step()
                record['path_seconds'] = durations
            if args.dispatch_census:
                from torch.utils._python_dispatch import TorchDispatchMode
                counters = {'update': Counter(), 'sentence': Counter()}
                waits, wait_calls = Counter(), Counter()
                sentence_seconds = 0.

                def measured_sentence(*a, **kw):
                    nonlocal sentence_seconds
                    start = time.perf_counter()
                    try:
                        return sentence(*a, **kw)
                    finally:
                        if not sentence_depth:
                            sentence_seconds += time.perf_counter() - start

                class DeviceReads(TorchDispatchMode):
                    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                        kwargs = kwargs or {}
                        name = func._schema.name
                        event = None
                        if torch.Tag.dynamic_output_shape in func.tags:
                            event = 'variable_shape/' + name
                        elif torch.Tag.data_dependent_output in func.tags:
                            event = 'host_result/' + name
                        elif name == 'aten::_to_copy' and kwargs.get('device') is not None:
                            if args[0].device.type != 'cpu' and torch.device(kwargs['device']).type == 'cpu':
                                event = 'device_to_host/' + name
                        elif name == 'aten::copy_' and args[0].device.type == 'cpu' and args[1].device.type != 'cpu':
                            event = 'device_to_host/' + name
                        if event is not None:
                            counters['update'][event] += 1
                            if sentence_depth:
                                counters['sentence'][event] += 1
                                if event.startswith(('host_result/', 'device_to_host/')):
                                    operands = args[1:] if name == 'aten::copy_' else args
                                    device = next((t.device.type for t in operands if torch.is_tensor(t)), 'unknown')
                                    key = f'{device}/{event}'
                                    start = time.perf_counter()
                                    result = func(*args, **kwargs)
                                    waits[key] += time.perf_counter() - start
                                    wait_calls[key] += 1
                                    return result
                        return func(*args, **kwargs)

                with patch.object(BasicModel, '_execute_input_sentence', measured_sentence), DeviceReads():
                    step()
                record['dispatch_census'] = counters
                # Includes queued device work waited for by these calls. No
                # extra synchronization is inserted in the measured sentence.
                record['host_dispatch_wall'] = dict(sentence_seconds=sentence_seconds,
                    seconds_by_source=waits, calls_by_source=wait_calls)
            if args.allocations:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                            profile_memory=True) as memory:
                    step()
                record['allocation_ops'] = {op.key: dict(calls=op.count, self_bytes=op.self_cpu_memory_usage)
                    for op in memory.key_averages() if op.self_cpu_memory_usage > 0 or 'resize' in op.key}
        finally:
            model.End()
    record['total_seconds'] = time.perf_counter() - start
    record['warmup_steps'] = record['steps'][:3]
    # Candidate walks share the sentence whose slot is counted above.
    steady_rows = [r for r in record['steps'][3:] if not r['graphs'] and not r['compiler_seconds'] and r['train']]
    steady = [r['seconds'] for r in steady_rows]
    record['steady_sentences'] = sum(r['sentences'] for r in steady_rows)
    record['steady_sentences_per_second'] = record['steady_sentences'] / sum(steady) if steady else None
    if steady_rows and 'end_and_reset_seconds' in steady_rows[0]:
        elapsed = sum(r['input_setup_seconds'] + r['seconds'] + r['end_and_reset_seconds'] for r in steady_rows)
        record['steady_sentences_per_second_with_boundaries'] = record['steady_sentences'] / elapsed
    record['steady_median_seconds'] = statistics.median(steady) if steady else None
    record['steady_mean_seconds'] = statistics.mean(steady) if steady else None
    save()
    print('finished', {k: v for k, v in record.items() if k.endswith('seconds')}, flush=True)


if __name__ == '__main__':
    main()
