"""Reproduce perf-1 measurements without choosing a learner seed.

Run with PYTHONPATH=bin:test BASICMODEL_DEVICE=cpu MODEL_COMPILE=none
TORCH_LOGS=graph_breaks,recompiles. Outputs are local evidence, not checkpoints.
Use --replay for matched sentence runs. XML capacities and batching stay intact.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
import time

import numpy as np
import torch


def write(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def leaves(args):
    from performance_leaves import cases, certify, evaluate
    results, values = {}, {}
    reference = torch.load(args.compare, weights_only=True) if args.compare else None
    for case in cases(device=args.device):
        def eager_values():
            expected = {}
            for batch in (3, 1, 2, 4, 8, 17):
                inputs = case.inputs(batch)
                for value in inputs:
                    if value.is_floating_point(): value.requires_grad_(True)
                expected[batch] = evaluate(case.function, inputs, case.parameters, gradients=True)
            return expected
        if args.record_eager:
            expected = eager_values()
            result = dict(recorded_eager=True)
        else:
            try:
                result, expected = certify(case, timing=True, require_clean=not args.audit_baseline)
            except Exception as error:
                if not args.audit_baseline:
                    raise
                result, expected = dict(certificate_error=repr(error)), eager_values()
        if reference is not None:
            torch.testing.assert_close(expected, reference[case.name], rtol=0, atol=0)
            result['eager_matches_baseline_bitwise'] = True
        results[case.name], values[case.name] = result, expected
        write(args.output/'leaves.json', results)
        print(case.name, json.dumps(result), flush=True)
    torch.save(values, args.output/'leaf-reference.pt')


def corpus(args):
    from prepare_corpus import materialize, build
    from prepare_data import prepare
    from Corpus import corpus_rows
    from data import Data
    root = Path.cwd()
    source = root/'data/fineweb/shard_00000.parquet'
    source_sha = digest(source)
    result = {}
    def measure(name, fn, repeats=5):
        seconds = []
        for _ in range(repeats):
            start = time.perf_counter(); value = fn(); seconds.append(time.perf_counter()-start)
        result[name] = dict(seconds=seconds, median_seconds=statistics.median(seconds))
        return value
    with contextlib.redirect_stdout(io.StringIO()), tempfile.TemporaryDirectory() as folder:
        temp = Path(folder)
        measure('verify_prepared_data', prepare)
        measure('materialize_sha256', lambda: materialize(
            dict(path=str(source.relative_to(root)), sha256=source_sha), root, root))
        entry = json.loads((root/'data/external-files.json').read_text())['files'][-1]
        entry = dict(entry, url=source.as_uri())
        index = 0
        def install():
            nonlocal index
            index += 1
            prepare(temp/f'plain-{index}', dict(files=[entry]))
        measure('prepare_plain_file_local_transport', install)
        recipe = temp/'recipe.json'
        recipe.write_text(json.dumps(dict(version=1, name='perf-fineweb-1000', groups=[dict(
            name='fineweb', adapter='document_parquet', max_documents=1000,
            sources=[dict(path=str(source.relative_to(root)), sha256=source_sha)])])))
        def assemble():
            nonlocal index
            index += 1
            destination = temp/f'corpus-{index}'
            build(recipe, destination, root=root)
            return destination
        folder = measure('prepare_corpus_1000_documents', assemble, repeats=3)
        result['shards'] = json.loads((folder/'manifest.json').read_text())['shards']
        rows = measure('load_corpus_rows', lambda: list(corpus_rows(folder/'manifest.json')))
        result['sentences'] = len(rows)
        result['rows_sha256'] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        data = Data()
        measure('load_curriculum', lambda: data.load('curriculum',
            dat={'corpusManifest': str(folder/'manifest.json')}, max_docs=None))
        result['loaded_sha256'] = hashlib.sha256(json.dumps(
            [data.train_input, data.validation_input, data.test_input, data.source_addresses], sort_keys=True).encode()).hexdigest()
    if args.compare:
        before = json.loads(args.compare.read_text())
        for key in ('shards', 'sentences', 'rows_sha256', 'loaded_sha256'):
            assert result[key] == before[key], key
        result['matches_baseline'] = True
    write(args.output/'corpus.json', result)
    print(json.dumps(result, indent=2))


def sentence(args):
    import cProfile
    import Language, util
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    from math_chain_corpus import ChainDocument, flatten
    from random_streams import capture
    # The existing eager mechanism: identical condition/body, no sentence capture.
    util.TheCompileBackend = 'none'
    def eager_while(condition, body, values):
        while bool(condition(*values)): values = body(*values)
        return values
    torch.while_loop = eager_while
    if args.replay:
        state = torch.load(args.replay/'initial-rng.pt', weights_only=False)
        torch.set_rng_state(state['torch'])
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])
    torch.save(capture(), args.output/'initial-rng.pt')
    config = Path('data')/('MM_corpus_lessons.xml' if args.kind == 'lesson' else 'MM_math_chain.xml')
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    start = time.perf_counter()
    model = BaseModel.from_config(str(config), data=TheData)[0]
    construction = time.perf_counter()-start
    model._tensor_peer_while_eager = True
    optimizer = model.getOptimizer(lr=.001)
    report = dict(kind=args.kind, config=str(config), config_sha256=digest(config),
        pythonhashseed=os.environ.get('PYTHONHASHSEED'),
        input_sha256=digest(Path('data/corpus-lessons.json') if args.kind == 'lesson'
                            else Path('data/math/grade-00.jsonl')),
        torch=str(torch.__version__), construction_seconds=construction,
        profiled=args.profile, records=[], presentations=[])
    run_batch = model.runBatch
    call_start = None
    def timed_batch(*positional, **keywords):
        nonlocal call_start
        call_start = time.perf_counter()
        return run_batch(*positional, **keywords)
    model.runBatch = timed_batch
    def observe(model, split, indices, result):
        elapsed = time.perf_counter()-call_start
        report['records'].append(dict(rows=indices, seconds=elapsed,
            loss_in=float(result.lossIn.detach()), loss_out=float(result.lossOut.detach()),
            breakdown=model.errors.breakdown(),
            answers=list(getattr(getattr(model, '_last_answer_construction', None), 'texts', ()) or ())))
        write(args.output/'progress.json', report)
        print(args.kind, len(report['records']), len(indices), elapsed, flush=True)
    profile = cProfile.Profile()
    def presentation(**metadata):
        start = time.perf_counter()
        if args.profile: profile.enable()
        try:
            summary = present(model, split='train', optimizer=optimizer,
                batch_size=1 if args.kind == 'lesson' else 8, after_batch=observe)
        finally:
            if args.profile: profile.disable()
        report['presentations'].append(dict(**metadata, seconds=time.perf_counter()-start, **summary))
    start = time.perf_counter()
    try:
        if args.kind == 'lesson':
            curriculum = json.loads(Path('data/corpus-lessons.json').read_text())
            for stage in curriculum['stages']:
                texts, targets = stage_documents(TheData, stage)
                admit_training_forms(model, texts, targets)
                for epoch in range(curriculum['epochs']):
                    presentation(stage=stage['name'], epoch=epoch)
        else:
            documents = [ChainDocument(**json.loads(line)['document']) for line in
                         Path('data/math/grade-00.jsonl').read_text().splitlines()]
            texts, targets, addresses = flatten(documents, supplied=True)
            TheData.train_input, TheData.train_output = texts, [torch.zeros(1) for _ in texts]
            TheData.text_answers = {'train': targets}
            TheData.has_supervised_outputs = any(target is not None for target in targets)
            TheData.source_addresses['train'] = [dict(address, split='train') for address in addresses]
            presentation(stage='frozen grade-00', epoch=0)
        report['status'] = 'complete'
    except Exception:
        import traceback
        report.update(status='failed', error=traceback.format_exc())
        print(report['error'], flush=True)
    finally:
        if args.profile: profile.dump_stats(args.output/'profile.pstats')
        report['total_seconds'] = time.perf_counter()-start
        torch.save({key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                    if torch.is_tensor(value)}, args.output/'final-tensors.pt')
        torch.save(optimizer.state_dict(), args.output/'final-optimizer.pt')
        torch.save(capture(), args.output/'final-rng.pt')
        write(args.output/'result.json', report)
        model.End()
    if args.compare:
        before = json.loads(args.compare.read_text())
        for key in ('kind', 'config_sha256', 'input_sha256', 'pythonhashseed'):
            if key in before:
                assert report[key] == before[key], key
        def observations(value):
            return [{k: v for k, v in row.items() if k != 'seconds'} for row in value['records']]
        assert observations(json.loads(json.dumps(report, default=str))) == observations(before)
        assert report['status'] == before['status']
        if report['status'] == 'failed':
            assert report['error'].splitlines()[-1] == before['error'].splitlines()[-1]
        tensors_before = args.compare.parent/'final-tensors.pt'
        if tensors_before.exists():
            torch.testing.assert_close(torch.load(args.output/'final-tensors.pt', weights_only=True),
                torch.load(tensors_before, weights_only=True), rtol=0, atol=0)
        optimizer_before = args.compare.parent/'final-optimizer.pt'
        if optimizer_before.exists():
            torch.testing.assert_close(torch.load(args.output/'final-optimizer.pt', weights_only=True),
                torch.load(optimizer_before, weights_only=True), rtol=0, atol=0)
        streams_before = args.compare.parent/'final-rng.pt'
        if streams_before.exists():
            previous = torch.load(streams_before, weights_only=False)
            current = torch.load(args.output/'final-rng.pt', weights_only=False)
            assert previous.keys() == current.keys()
            for key in previous:
                if key == 'numpy':
                    assert previous[key][0] == current[key][0]
                    assert np.array_equal(previous[key][1], current[key][1])
                    assert previous[key][2:] == current[key][2:]
                elif key == 'python':
                    assert previous[key] == current[key]
                else:
                    torch.testing.assert_close(previous[key], current[key], rtol=0, atol=0)
        report['matches_baseline_bitwise'] = True
        write(args.output/'result.json', report)
    print(json.dumps({key: report[key] for key in ('kind', 'status', 'construction_seconds', 'total_seconds')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    commands = parser.add_subparsers(dest='command', required=True)
    leaf = commands.add_parser('leaves')
    leaf.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    leaf.add_argument('--record-eager', action='store_true')
    leaf.add_argument('--audit-baseline', action='store_true')
    commands.add_parser('corpus')
    model = commands.add_parser('sentence')
    model.add_argument('kind', choices=('lesson', 'math'))
    model.add_argument('--replay', type=Path)
    model.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
