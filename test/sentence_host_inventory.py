"""Observe native lesson work, without changing fixtures or selecting a seed.

Public tensor calls, dispatcher operations and explicit host-read APIs are
separate counters (never summed together). Detailed mode additionally counts
executed Python loop iterations and profiles one complete lesson sentence.
Loop instrumentation exists only in this test process's import loader.
"""
import argparse
import ast
import builtins
from contextlib import nullcontext
from collections import Counter
import cProfile
import importlib.abc
import importlib.machinery
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from torch.overrides import TorchFunctionMode
from torch.utils._python_dispatch import TorchDispatchMode

from attention_thinking_certificate import frozen, parameters, restore, stream, stream_digest


class Inventory:
    def __init__(self, source, detail, profile=False):
        self.source = str(source.resolve()) + os.sep
        self.detail = detail
        self.profile = profile
        self.active = False
        self.api, self.ops, self.host = Counter(), Counter(), Counter()
        self.host_sites, self.loop_entries, self.loop_iterations = Counter(), Counter(), Counter()
        self.loop_sites = {}

    def site(self):
        frame = inspect.currentframe().f_back
        site = None
        phase = None
        while frame is not None:
            code = frame.f_code
            if code.co_filename.startswith(self.source):
                name = code.co_qualname
                module = Path(code.co_filename).stem
                if site is None:
                    site = (module + '.py', name, frame.f_lineno)
                if phase is None:
                    if 'thought' in name.lower() or module.startswith('Thought'):
                        phase = 'thought'
                    elif ('generat' in name.lower() or 'reconstruct' in name.lower()
                          or module in ('Generative', 'FieldReverse', 'SentenceReverse')):
                        phase = 'generate walks'
                    elif module == 'Attention':
                        phase = 'walk'
                    elif any(part in name for part in ('restore_sentence', 'recall_sentence',
                                                       '_snapshot_sentence_state', '_publish_sentence_scratch')):
                        phase = 'restoration'
                    elif 'word_step' in name or name.endswith(('compose_word', 'read_field')):
                        phase = 'word step'
                    elif ('_commit_sentence' in name or module in ('SentenceClose', 'SentenceClosing')
                          or module == 'WholePaths' and code.co_name in ('finish', 'keep_output')
                          or module == 'SentenceTraining' and code.co_name == 'finish_sentence'):
                        phase = 'close'
            frame = frame.f_back
        return (phase or 'preparation', *(site or ('external', 'external', 0)))

    def loop_iter(self, key, value):
        if not self.active:
            return value
        group = (self.site()[0], key)
        self.loop_entries[group] += 1
        def values():
            for item in value:
                if self.active:
                    self.loop_iterations[group] += 1
                yield item
        return values()

    def loop_test(self, key, value):
        result = bool(value)
        if self.active:
            group = (self.site()[0], key)
            self.loop_entries[group] += 1
            self.loop_iterations[group] += int(result)
        return result

    def totals(self):
        return dict(tensor_api_calls=sum(self.api.values()), dispatcher_calls=sum(self.ops.values()),
                    host_read_api_calls=sum(self.host.values()), host_read_apis=dict(self.host),
                    scalar_dispatches=self.ops.get('aten._local_scalar_dense.default', 0),
                    dynamic_nonzero_dispatches=self.ops.get('aten.nonzero.default', 0))


HOST_APIS = {'tolist', 'cpu', 'item', 'nonzero', 'numpy', '__bool__', '__int__', '__index__', '__float__'}


class ApiCalls(TorchFunctionMode):
    def __init__(self, inventory):
        self.inventory = inventory

    def __torch_function__(self, func, types, args=(), kwargs=None):
        seen = self.inventory
        if seen.active:
            name = getattr(func, '__name__', str(func))
            seen.api[name] += 1
            if name in HOST_APIS:
                seen.host[name] += 1
                if seen.detail:
                    seen.host_sites[(*seen.site(), name)] += 1
        return func(*args, **(kwargs or {}))


class DispatcherCalls(TorchDispatchMode):
    def __init__(self, inventory):
        self.inventory = inventory

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.inventory.active:
            self.inventory.ops[str(func)] += 1
        return func(*args, **(kwargs or {}))


class Loops(ast.NodeTransformer):
    def __init__(self, inventory, filename):
        self.inventory, self.filename, self.scope = inventory, filename, []

    def visit_FunctionDef(self, node):
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef

    def call(self, node, kind, value, method):
        key = f'{Path(self.filename).name}:{node.lineno}:{node.col_offset}:{kind}'
        self.inventory.loop_sites[key] = dict(file=Path(self.filename).name,
            function='.'.join(self.scope), line=node.lineno, kind=kind,
            expression=ast.unparse(value))
        return ast.copy_location(ast.Call(ast.Attribute(ast.Name('_host_inventory_runtime', ast.Load()),
            method, ast.Load()), [ast.Constant(key), value], []), value)

    def visit_For(self, node):
        self.generic_visit(node)
        node.iter = self.call(node, 'for', node.iter, 'loop_iter')
        return node

    def visit_While(self, node):
        self.generic_visit(node)
        node.test = self.call(node, 'while', node.test, 'loop_test')
        return node

    def visit_comprehension(self, node):
        self.generic_visit(node)
        node.iter = self.call(node.iter, 'comprehension', node.iter, 'loop_iter')
        return node


class LoopLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        tree = ast.parse(self.get_source(fullname), filename=self.path)
        tree = Loops(builtins._basicmodel_host_inventory, self.path).visit(tree)
        at = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else 0
        while at < len(tree.body) and isinstance(tree.body[at], ast.ImportFrom) and tree.body[at].module == '__future__':
            at += 1
        tree.body.insert(at, ast.ImportFrom('builtins', [ast.alias('_basicmodel_host_inventory', '_host_inventory_runtime')], 0))
        return compile(ast.fix_missing_locations(tree), self.path, 'exec')


class LoopFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec and spec.origin and spec.origin.startswith(builtins._basicmodel_host_inventory.source) and spec.origin.endswith('.py'):
            spec.loader = LoopLoader(fullname, spec.origin)
            return spec


class ObservationComplete(Exception):
    pass


def compare_states(output, reference):
    """Require exact model, optimizer and RNG records at every observed batch."""
    batches = sorted(output.glob('batch-*.json'))
    if not batches:
        raise AssertionError('no observed batch states')
    for path in batches:
        actual = json.loads(path.read_text())
        expected = json.loads((reference/path.name).read_text())
        changed = [key for key in actual.keys() | expected.keys()
                   if actual.get(key) != expected.get(key)]
        if changed:
            raise AssertionError(f'{path.name}: changed {changed}')
    return len(batches)


class StateObserver:
    """Only record completed batches, outside the measured sentence window."""
    def __init__(self, output):
        self.output, self.records = output, []
        self.reference_seconds = 0.
        self.calls, self.actions, self.costs = [], [], []

    def batch(self, model, *, seconds, label, optimizer=None):
        payload = dict(label=label, parameters=parameters(model), stream=stream_digest())
        if optimizer is not None:
            payload['optimizer'] = frozen(optimizer.state_dict())
        index = len(self.records)
        (self.output/f'batch-{index:03}.json').write_text(json.dumps(payload)+'\n')
        self.records.append(dict(batch=index, label=label, seconds=seconds))


class ProfileObserver:
    def __init__(self, patch, output, inventory):
        import Attention, ModelAttention, SentenceTraining
        self.output, self.inventory = output, inventory
        self.records, self.calls, self.actions, self.costs, self.walks = [], [], [], [], []
        self.reference_seconds = 0.
        self.profile = cProfile.Profile() if inventory.detail or inventory.profile else None
        stage = ModelAttention.stage_input
        def start(model):
            inventory.active = True
            if self.profile:
                self.profile.enable()
            return stage(model)
        patch.setattr(ModelAttention, 'stage_input', start)
        finish = SentenceTraining.finish_sentence
        def close(model):
            result = finish(model)
            inventory.active = False
            if self.profile:
                self.profile.disable()
            return result
        patch.setattr(SentenceTraining, 'finish_sentence', close)
        opening = Attention.open_narrowing
        self.latest = None
        def opened(*args, **kwargs):
            self.latest = opening(*args, **kwargs)
            return self.latest
        patch.setattr(Attention, 'open_narrowing', opened)
        patch.setattr(ModelAttention, 'open_narrowing', opened)
        read = Attention.narrow_words
        def reading(*args, **kwargs):
            result = read(*args, **kwargs)
            self.walks.append(dict(owner='staging-or-nonsentence', rounds=self.latest.steps_taken))
            return result
        patch.setattr(Attention, 'narrow_words', reading)
        patch.setattr(ModelAttention, 'narrow_words', reading)
        restore_read = ModelAttention.SentenceNarrowing.for_restore
        def restoring(walk, index):
            result = restore_read(walk, index)
            self.walks.append(dict(owner='sentence', rounds=walk.state.steps_taken,
                                  departure=walk.state.departure is not None))
            return result
        patch.setattr(ModelAttention.SentenceNarrowing, 'for_restore', restoring)

    def batch(self, model, *, seconds, label, optimizer=None):
        self.inventory.active = False
        if self.profile:
            self.profile.disable()
        total = self.inventory.totals()
        previous = self.records[-1]['cumulative'] if self.records else {}
        count = {k: v-previous.get(k, 0) for k, v in total.items() if isinstance(v, int)}
        row = dict(batch=len(self.records), label=label, seconds=seconds,
                   counts=count, cumulative=total, walks=self.walks)
        self.records.append(row)
        self.walks = []
        payload = dict(label=label, parameters=parameters(model), stream=stream_digest())
        if optimizer is not None:
            payload['optimizer'] = frozen(optimizer.state_dict())
        (self.output/f'batch-{row["batch"]:03}.json').write_text(json.dumps(payload)+'\n')
        print(json.dumps(row), flush=True)
        if self.inventory.detail or self.inventory.profile:
            raise ObservationComplete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--detail', action='store_true')
    parser.add_argument('--profile', action='store_true', help='cProfile alone on the first complete sentence')
    parser.add_argument('--plain', action='store_true', help='unobserved reference, recorded only after each batch')
    parser.add_argument('--compare', type=Path, help='require exact state equality to an unobserved run')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    assert sum((args.detail, args.profile, args.plain)) <= 1
    inventory = Inventory(args.source/'bin', args.detail, args.profile)
    builtins._basicmodel_host_inventory = inventory
    if args.detail:
        sys.meta_path.insert(0, LoopFinder())
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    from attention_thinking_certificate import lesson
    torch.set_num_threads(1)
    with MonkeyPatch.context() as patch:
        use_eager_reading(patch)
        observer = StateObserver(args.output) if args.plain else ProfileObserver(patch, args.output, inventory)
        metadata = json.loads((args.replay/'initial.json').read_text())
        assert os.environ.get('PYTHONHASHSEED') == metadata['pythonhashseed']
        restore(torch.load(args.replay/'initial-rng.pt', weights_only=False))
        started = time.perf_counter()
        with (nullcontext() if args.profile or args.plain else ApiCalls(inventory)), \
             (nullcontext() if args.profile or args.plain else DispatcherCalls(inventory)):
            try:
                lesson(observer)
            except ObservationComplete:
                pass
        report = dict(source_commit=subprocess.check_output(
                          ['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip(),
                      seconds=time.perf_counter()-started, records=observer.records,
                      counts=inventory.totals(), tensor_apis=dict(inventory.api),
                      dispatcher_operations=dict(inventory.ops),
                      host_sites=[dict(phase=p,file=f,function=n,line=l,api=a,count=v)
                                  for (p,f,n,l,a),v in inventory.host_sites.items()],
                      loops=[dict(phase=p, **inventory.loop_sites[key], entries=n,
                                  iterations=inventory.loop_iterations[(p,key)])
                             for (p,key),n in inventory.loop_entries.items()])
        if getattr(observer, 'profile', None):
            observer.profile.dump_stats(str(args.output/'sentence.pstats'))
        if args.compare:
            report['equal_batches'] = compare_states(args.output, args.compare)
        (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report['counts']),flush=True)


if __name__ == '__main__':
    main()
