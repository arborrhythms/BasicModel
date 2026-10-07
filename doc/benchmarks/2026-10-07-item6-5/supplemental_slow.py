"""Run the four separately required legacy slow checks once, after the campaign."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(ROOT/'test'), str(HERE)]
import bounded_tests as bounded
from campaign import environment
from verification import validate


def run():
    complete = json.loads((HERE/'measurements/complete.json').read_text())
    assert complete['completed'] and complete['source_matched']
    source = bounded.source_snapshot(ROOT)
    validate(source)
    protocol = json.loads((HERE/'supplemental-slow-protocol.json').read_text())
    helpers = json.loads((HERE/'supplemental-helper-manifest.json').read_text())
    out = HERE/'supplemental-slow'
    out.mkdir(exist_ok=False)
    records = []
    for number, selector in enumerate(protocol['selectors'], 1):
        assert source == bounded.source_snapshot(ROOT)
        assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == sha
                   for name, sha in helpers.items())
        folder = out/f'case-{number:02}'
        folder.mkdir()
        env = environment()
        env.update(PYTEST_PLUGINS='operators_gate_observer', ITEM7_XOR_GATE='0',
                   ITEM7_XOR_MEASUREMENTS=str(folder/'observations.jsonl'),
                   REVIEW17_REPORTS=str(folder/'reports.jsonl'))
        process = bounded.GuardedProcess([sys.executable, '-m', 'pytest', '-q', selector],
            cwd=ROOT, env=env, log_path=folder/'run.log', memory_bytes=8*bounded.GIB,
            timeout=1800).start()
        try:
            result = None
            while result is None:
                result = process.poll()
                if result is None:
                    time.sleep(.5)
        finally:
            if process.proc.poll() is None:
                process.stop(exit_code=130, reason='supplement_stopped')
        bounded.write_json(folder/'process.json', result)
        records.append(dict(selector=selector, folder=str(folder.relative_to(HERE)), process=result))
        bounded.write_json(out/'progress.json', records)
        print(json.dumps(dict(selector=selector, exit_code=result['exit_code'], reason=result['reason'])), flush=True)
    bounded.write_json(out/'complete.json', dict(records=records,
        source_matched=source == bounded.source_snapshot(ROOT),
        source=source, seed=None, retries=0, replacement_trainings=0,
        scope='Four legacy slow cases required by doc/Testing.md; separate from the thirty standing trainings.'))


if __name__ == '__main__':
    run()
