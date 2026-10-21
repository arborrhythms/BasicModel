"""Capture the actual initialization and failure read of the unchanged certificate.

No model reads are added during execution. Working values below are the values
returned by reads that the certificate/model already performs.
"""
import hashlib
import json
import os
from pathlib import Path
import random
import shutil

import numpy as np
import pytest
import torch


def rng():
    return dict(torch=torch.get_rng_state(), python=random.getstate(),
                numpy=np.random.get_state())


def restore(value):
    torch.set_rng_state(value['torch'])
    random.setstate(value['python'])
    np.random.set_state(value['numpy'])


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [plain(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def meaning(value):
    from ThoughtReferences import bindings, open_slots
    if value is None:
        return None
    return dict(refs=value.role_refs, bindings=bindings(value), open=open_slots(value),
                roles=value.roles, mask=value.role_mask)


def snapshot(model):
    owner = model._concept_owner()
    alloc, store = owner._concept_allocator, model.symbolSpace.ltm_store
    return dict(
        rows=[dict(index=i, address=store.occurrence_of(i),
                   document=store.document_keys[i], content=store.sentence_content_keys[i],
                   kind=store.KINDS[int(store.record_kind[i])],
                   stream=store.stream[i], raw=meaning(store.raw_meaning_of(i)))
              for i in range(len(store))],
        mtm={str(key): vars(entry) for key, entry in owner.mtm.entries.items()},
        words=alloc.word_forms, types=alloc.symbol_types,
        singletons=sorted(alloc.singletons),
        occupations=dict(model.languageSpace.parts_of_speech.counts),
        parts={int(i):store.parts.row(i) for i in store.parts.indices().tolist()},
        parameters={name:tensor_hash(value) for name,value in model.named_parameters()},
        sources=getattr(model, '_sentence_sources', None),
        source_manifest=model.inputSpace.data.source_manifest,
        rng=tensor_hash(torch.get_rng_state()))


@pytest.fixture(autouse=True)
def capture_arrival(request, monkeypatch):
    if 'test_forced_ordinary_pending_premise_in_both_orders' not in request.node.name:
        return
    import math_chain_ordinary
    from ClauseRow import ClauseRows
    baseline = os.environ.get('ARRIVAL_CLOSING_BASELINE')
    if baseline:
        import importlib.util
        import PartClosing
        spec = importlib.util.spec_from_file_location('diagnostic_closing_baseline', baseline)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        monkeypatch.setattr(PartClosing, 'PartClosing', module.PartClosing)
    directory = Path(os.environ['ARRIVAL_CAPTURE']) / request.node.name
    directory.mkdir(parents=True, exist_ok=False)
    request.node._arrival_directory = directory

    def write(name, value):
        (directory/name).write_text(json.dumps(plain(value), indent=2)+'\n')

    # The generator owns a private Random(None), independent of the three
    # process RNGs. Its presentation digest participates in source addresses.
    from exact import MathProblemGenerator
    generator_init = MathProblemGenerator.__init__
    generator_count = 0
    def construct_generator(generator, *args, **kwargs):
        nonlocal generator_count
        generator_init(generator, *args, **kwargs)
        name = f'generator-{generator_count}.pt'
        generator_count += 1
        replay = os.environ.get('ARRIVAL_REPLAY')
        prior = None if replay is None else Path(replay)/name
        if prior is not None and prior.exists():
            generator.rng.setstate(torch.load(prior, weights_only=False))
        torch.save(generator.rng.getstate(), directory/name)
    monkeypatch.setattr(MathProblemGenerator, '__init__', construct_generator)

    original = math_chain_ordinary.ordinary_model
    def construct(*args, **kwargs):
        replay = os.environ.get('ARRIVAL_REPLAY')
        if replay:
            restore(torch.load(Path(replay)/'initial-rng.pt', weights_only=False))
        torch.save(rng(), directory/'initial-rng.pt')
        model = original(*args, **kwargs)
        if replay and not (Path(replay)/'generator-0.pt').exists():
            # Older captures retained the realized manifest rather than the
            # private generator. The test replaces the generated documents;
            # retain their captured source digest, the remaining used state.
            prior = json.loads((Path(replay)/'initial-model.json').read_text())
            if 'source_manifest' in prior:
                model.inputSpace.data.source_manifest = prior['source_manifest']
        write('initial-model.json', dict(
            parameters={name:tensor_hash(value) for name,value in model.named_parameters()},
            buffers={name:tensor_hash(value) for name,value in model.named_buffers()},
            after_rng=tensor_hash(torch.get_rng_state()),
            closing_baseline=baseline,
            source_manifest=model.inputSpace.data.source_manifest,
            python_hash_probe=hash('arrival-capture'),
            torch_threads=torch.get_num_threads()))
        torch.save(rng(), directory/'after-model-rng.pt')
        request.node._arrival_model = model
        return model
    monkeypatch.setattr(math_chain_ordinary, 'ordinary_model', construct)

    read = ClauseRows.read_reference_bindings
    def observe_read(store, index, value, **kwargs):
        result = read(store, index, value, **kwargs)
        from ThoughtReferences import bindings
        if bindings(value).get('_forward_references'):
            with (directory/'reads.jsonl').open('a') as stream:
                stream.write(json.dumps(plain(dict(index=index, before=meaning(value),
                    after=meaning(result), exclude=kwargs.get('exclude', ()))))+'\n')
        return result
    monkeypatch.setattr(ClauseRows, 'read_reference_bindings', observe_read)

    train = math_chain_ordinary.train_documents
    def observed_train(model, documents, folder, *, after=None, **kwargs):
        batch = 0
        def observed_after(*args):
            nonlocal batch
            batch += 1
            try:
                return after(*args) if after is not None else None
            except BaseException:
                # Freeze the native rows before the test's finally/End.
                write('failure-store.json', snapshot(model))
                write('failure-batch.json', dict(batch=batch, observations=args[4]))
                from PartRead import expression_components
                from QueryWork import QueryWorkBudget
                import sys
                trace = sys.exc_info()[2]
                while trace:
                    local = trace.tb_frame.f_locals
                    if trace.tb_frame.f_code.co_name in ('after', 'check') and 'value' in local:
                        value = local['value']
                        roots = tuple(ref for ref in value.role_refs
                                      if ref is not None and ref[0] in ('sym', 'ltm'))
                        terms, scanned, incomplete = expression_components(
                            model.symbolSpace.ltm_store, roots, max_records=256,
                            work=QueryWorkBudget(256))
                        write('failure-expression.json', dict(scanned=scanned,
                            incomplete=incomplete, terms={str(k):dict(
                                kind=v.get('kind'), reference=v.get('reference'),
                                operands=v.get('operands')) for k,v in terms.items()}))
                    trace = trace.tb_next
                raise
        return train(model, documents, folder, after=observed_after, **kwargs)
    monkeypatch.setattr(math_chain_ordinary, 'train_documents', observed_train)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    directory = getattr(item, '_arrival_directory', None)
    if directory is None or report.when != 'call':
        return
    details = dict(node=item.nodeid, outcome=report.outcome, failure=str(report.longrepr))
    if call.excinfo is not None:
        details['frames'] = [dict(name=frame.name, line=frame.lineno,
            values={key: meaning(value) if key == 'value' else plain(value)
                    for key,value in frame.frame.f_locals.items()
                    if key in ('index', 'address', 'value', 'correct', 'pending', 'reverse')})
            for frame in call.excinfo.traceback if frame.name in (
                'after', 'check', 'test_forced_ordinary_pending_premise_in_both_orders')]
    (directory/'outcome.json').write_text(json.dumps(plain(details), indent=2)+'\n')
    # Copy the native question tables only after the test has finished. This
    # adds no model reads or changes to the trial that produced the result.
    folder = item.funcargs.get('tmp_path')
    if folder is not None:
        for name in ('certificate.json', 'ordinary.json'):
            source = folder/name
            if source.is_file():
                shutil.copy2(source, directory/name)
