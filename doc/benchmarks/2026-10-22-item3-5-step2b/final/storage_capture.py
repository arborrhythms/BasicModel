"""Observe the contract units without changing their values or choices."""
import json
import os
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from arrival_capture import plain


@pytest.fixture(autouse=True)
def capture_contract(request, monkeypatch):
    if 'test_forced_ordinary_pending_premise' in request.node.name:
        return
    root = Path(os.environ['CONTRACT_CAPTURE']) / request.node.nodeid.replace('/', '_').replace('::', '--')
    root.mkdir(parents=True, exist_ok=False)
    torch.save(dict(torch=torch.get_rng_state(), python=random.getstate(),
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
            torch.save(dict(parameters=resolver.components.owner.state_dict(),
                            store=resolver.store.state_dict()), root/f'initial-{len(resolvers)}.pt')
        result = preview(resolver, *args, **kwargs)
        records.append(dict(call='preview', plan=plain({
            key: result.get(key) for key in ('items', 'individuals', 'arrivals', 'decisions', 'cost')})))
        return result

    monkeypatch.setattr(ClauseJournal, 'finish_clause', journal)
    monkeypatch.setattr(ReferenceClose, 'preview', close)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    result = yield
    report = result.get_result()
    root = getattr(item, '_contract_capture', None)
    if root is None or report.when != 'call':
        return
    (root/'outcome.json').write_text(json.dumps(dict(node=item.nodeid,
        outcome=report.outcome, failure=str(report.longrepr),
        records=item._contract_records), indent=2)+'\n')
    if report.failed:
        for index, resolver in enumerate(item._contract_resolvers, 1):
            torch.save(dict(parameters=resolver.components.owner.state_dict(),
                store=resolver.store.state_dict(), mtm=resolver.components.owner.mtm.entries,
                audit=resolver.last_audit), root/f'failure-{index}.pt')
