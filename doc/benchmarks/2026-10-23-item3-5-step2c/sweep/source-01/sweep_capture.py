"""Observe the contract units without changing their values or choices."""
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from arrival_capture import plain
from pack_capture import save


def observed_state(resolver):
    try:
        return memory_state(resolver)
    except Exception as error:
        return dict(capture_error=type(error).__name__, message=str(error))


def memory_state(resolver):
    owner = resolver.components.owner
    model = getattr(owner, '_model', None)
    structural = getattr(model, '_collect_structural_extras', None)
    from Spaces import _concept_alloc_of
    def optional(read):
        try:
            return None if read is None else read()
        except Exception as error:
            # A corruption fixture may deliberately make a checkpoint invalid.
            # Record that limitation without changing the call under test.
            return dict(capture_error=type(error).__name__, message=str(error))
    return dict(parameters=owner.state_dict(), store=resolver.store.state_dict(),
        store_semantics=optional(resolver.store.semantic_extras),
        parts=_concept_alloc_of(owner).parts.state_dict(),
        owner_extras=optional(owner.vocab_extras), mtm_state=owner.mtm.state(),
        model_structural=optional(structural))


@pytest.fixture(autouse=True)
def capture_contract(request, monkeypatch):
    # Nested harness probes in temporary roots do not belong to this sweep.
    if not Path(request.node.path).resolve().is_relative_to(Path(__file__).resolve().parents[5] / "test"):
        return
    if 'test_forced_ordinary_pending_premise' in request.node.name:
        return
    replay = os.environ.get('IDENTITY_REPLAY_RNG')
    if replay:
        from arrival_capture import restore
        restore(torch.load(replay, weights_only=False))
    root = Path(os.environ['CONTRACT_CAPTURE']) / request.node.nodeid.replace('/', '_').replace('::', '--')
    attempt = 1
    original_root = root
    while root.exists():
        attempt += 1
        root = original_root.with_name(original_root.name + f"--attempt-{attempt}")
    root.mkdir(parents=True, exist_ok=False)
    save(dict(torch=torch.get_rng_state(), python=random.getstate(),
                    numpy=np.random.get_state()), root/'initial-rng.pt')
    request.node._contract_capture = root
    # Corpus generators own private RNGs. Retain them as well as the process
    # RNG so a failing driver can reproduce its actual presentation.
    from exact import MathProblemGenerator
    generator_init, generator_count = MathProblemGenerator.__init__, 0
    def generator(generator, *args, **kwargs):
        nonlocal generator_count
        generator_init(generator, *args, **kwargs)
        name = f'generator-{generator_count}.pt'
        generator_count += 1
        replay = os.environ.get('IDENTITY_REPLAY_GENERATORS')
        if replay:
            generator.rng.setstate(torch.load(Path(replay)/name, weights_only=False))
        save(generator.rng.getstate(), root/name)
    monkeypatch.setattr(MathProblemGenerator, '__init__', generator)
    records = request.node._contract_records = []
    resolvers = request.node._contract_resolvers = []
    import ClauseJournal
    from ReferenceClose import ReferenceClose
    finish, preview = ClauseJournal.finish_clause, ReferenceClose.preview

    def journal(language, program, **kwargs):
        records.append(dict(call='journal', program=plain(vars(program))))
        result = finish(language, program, **kwargs)
        from storage_probe import clause_graph
        records.append(dict(call='finished', clause=clause_graph(result)))
        return result

    def close(resolver, *args, **kwargs):
        if all(resolver is not previous for previous in resolvers):
            resolvers.append(resolver)
            save(observed_state(resolver), root/f'initial-{len(resolvers)}.pt')
        result = preview(resolver, *args, **kwargs)
        records.append(dict(call='preview', plan=plain({
            key: result.get(key) for key in ('items', 'individuals', 'arrivals', 'decisions', 'cost')})))
        return result

    monkeypatch.setattr(ClauseJournal, 'finish_clause', journal)
    monkeypatch.setattr(ReferenceClose, 'preview', close)
    import ThoughtStream
    from arrival_capture import meaning
    integrate = ThoughtStream.integrate
    def thought(registry, goal, result, prior):
        value = integrate(registry, goal, result, prior)
        records.append(dict(call='thought-integrate', goal=plain(meaning(goal)),
            operation=None if result is None else result.semantic_id,
            supplied=None if result is None else plain(meaning(result.evidence.get('meaning'))),
            frames=[] if result is None else [dict(occurrence=frame.get('occurrence'),
                meaning=plain(meaning(frame.get('meaning')))) for frame in result.evidence.get('frames', ())],
            updated=plain(meaning(value))))
        return value
    monkeypatch.setattr(ThoughtStream, 'integrate', thought)



@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    root = getattr(item, '_contract_capture', None)
    if root is None or report.when != 'call':
        return
    encoded = (json.dumps(dict(node=item.nodeid,
        outcome=report.outcome, failure=str(report.longrepr),
        records=item._contract_records), indent=2)+'\n').encode()
    if len(encoded) > 2 * 1024 * 1024:
        import gzip
        (root/'outcome.json.gz').write_bytes(gzip.compress(encoded, mtime=0))
    else:
        (root/'outcome.json').write_bytes(encoded)
    if report.failed:
        import shutil
        temporary = item.funcargs.get('tmp_path')
        if temporary is not None:
            shutil.copytree(temporary, root/'artifacts', dirs_exist_ok=False)
        for index, resolver in enumerate(item._contract_resolvers, 1):
            save(dict(observed_state(resolver), mtm=getattr(getattr(resolver.components.owner, 'mtm', None), 'entries', None),
                audit=getattr(resolver, 'last_audit', None)), root/f'failure-{index}.pt')
