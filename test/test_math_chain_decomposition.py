"""Forced architecture demonstrations, with every companion-row outcome kept."""
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.mark.slow
@pytest.mark.parametrize('successors', (1, 2))
def test_forced_native_decomposition_through_training_driver(tmp_path, successors):
    path = Path(__file__).resolve().parents[1]/'doc/benchmarks/2026-10-08-math-chain-repair-2/decomposition_certificate.py'
    spec = importlib.util.spec_from_file_location('forced_decomposition_certificate', path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    folder = tmp_path/f'decomposition-{successors}'
    report = fixture.run(folder, successors)
    questions = [json.loads(line) for line in (folder/'questions.jsonl').read_text().splitlines()]
    # This is an existence demonstration of the architecture, not an
    # accuracy bar. A live compose departure may keep a different reading;
    # every companion row is reported, including those without this proof.
    proved = [question for question in questions if question['chain_correct']]
    assert proved, questions
    assert len(questions) >= 2
    assert len({row['departure'] for row in report['rows'] if row['departure'] >= 0}) > 1
    for question in proved:
        episode = report['episodes'][int(question['document'])]
        assert question['bound_correct'] and question['inference_count'] == successors
        assert sum(record['kind'] == 'descend' for record in episode['trace']) == successors
        assert sum(record['kind'] == 'return' for record in episode['trace']) == successors
        assert episode['trace'][-2]['operation'] == 'conclude'
        assert episode['state_diff']['unexpected'] == []
        assert episode['state_diff']['added_row_kinds'] == ['inference'] * successors
