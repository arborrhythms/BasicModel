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
    if 'test_forced_ordinary_pending_premise' in request.node.name:
        return
    replay = os.environ.get('IDENTITY_REPLAY_RNG')
    if replay:
        from arrival_capture import restore
        restore(torch.load(replay, weights_only=False))
    root = Path(os.environ['CONTRACT_CAPTURE']) / request.node.nodeid.replace('/', '_').replace('::', '--')
    root.mkdir(parents=True, exist_ok=False)
    save(dict(torch=torch.get_rng_state(), python=random.getstate(),
                    numpy=np.random.get_state()), root/'initial-rng.pt')
    request.node._contract_capture = root
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
            save(memory_state(resolver), root/f'initial-{len(resolvers)}.pt')
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

    from PartClosing import PartClosing
    closing = PartClosing.close
    def close_part(owner, clause, **kwargs):
        result = closing(owner, clause, **kwargs)
        if clause.relation is not None and (result is None or not owner.store.parts.contains(result)):
            records.append(dict(call='missing-part-edge', result=result,
                same_parts=owner.parts is owner.store.parts,
                in_parts=False if result is None else owner.parts.contains(result),
                plan=[dict(operation=entry['value'].operation, symbol=entry['symbol'],
                    edge=entry['edge'], pair=entry['pair'], seat=entry['seat'],
                    children=[owner._planned[id(child)]['symbol'] for child in entry['value'].children])
                    for entry in owner._plan]))
        return result
    monkeypatch.setattr(PartClosing, 'close', close_part)


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
            save(dict(memory_state(resolver), mtm=resolver.components.owner.mtm.entries,
                audit=resolver.last_audit), root/f'failure-{index}.pt')
