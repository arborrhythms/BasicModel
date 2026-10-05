"""Record the unchanged-seed ordinary chooser batch, including failed probes."""
import json
import os
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def observe_chooser_test(request):
    if 'test_normal_text_reconstruction_updates_the_grammar_chooser' not in request.node.name:
        yield
        return
    from review16_score_probe import observe_score_function
    from review16_run_audit import json_value
    result = {}
    try:
        with observe_score_function(result):
            yield
    finally:
        destination = Path(os.environ['REVIEW16_TEST_AUDIT'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('a') as f:
            f.write(json.dumps(dict(node=request.node.nodeid, audit=result), default=json_value)+'\n')
