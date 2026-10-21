"""Observe the unchanged partition assertion under the decided exclusion rule."""
import json
from pathlib import Path
import pytest
from arrival_capture import plain


@pytest.fixture(autouse=True)
def observe_partition(request, monkeypatch):
    if 'test_same_kind_fourth_close_preserves_mints_and_revises_ambiguous_binding' not in request.node.name:
        return
    import test_reference_close as tests
    from ReferenceClose import ReferenceClose
    observed = []
    close, read = ReferenceClose.preview, tests.current_identities
    def preview(resolver, *args, **kwargs):
        result = close(resolver, *args, **kwargs)
        observed.append(plain(dict(kind='preview', items=[
            {key: item.get(key) for key in ('mode', 'content', 'predicate', 'selected',
             'candidates', 'features')} for item in result['items']])))
        return result
    def identities(resolver, frames):
        result = read(resolver, frames)
        observed.append(plain(dict(kind='partition', identities=result,
            pool={str(key): vars(value) for key, value in resolver.components.owner.mtm.entries.items()})))
        return result
    monkeypatch.setattr(ReferenceClose, 'preview', preview)
    monkeypatch.setattr(tests, 'current_identities', identities)
    yield
    (Path(__file__).parent/'partition-diagnostic.json').write_text(json.dumps(
        dict(test=request.node.nodeid, observations=observed), indent=2)+'\n')
