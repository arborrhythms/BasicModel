"""Execute actual B/C/D controls after detecting a no-op diagnostic override.

The first diagnostics are retained in bisections/ and explicitly invalidated.
This helper uses XMLConfig's public setter on the live singleton, reapplies
after each load/overlay, and asserts the instantiated index's effective values.
Standing measurements are never changed or replaced.
"""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / 'verified-bisections'
sys.path[:0] = [str(HERE), str(ROOT / 'bin'), str(ROOT / 'test')]


@contextmanager
def effective_switch(part, folder):
    from unittest.mock import patch
    from util import XMLConfig, TheXMLConfig
    from MereologicalCodes import MereologicalCodes
    key, value = {'B': ('meaningWidth', 0), 'C': ('symbolCentroid', False),
                  'D': ('membershipPriming', False)}[part]
    log = dict(part=part, key=key, value=value, loads=[], instances=[])
    def apply(config, source):
        config.set('ConceptualSpace.' + key, value)
        actual = config.space('ConceptualSpace', key)
        assert type(actual) is type(value) and actual == value
        log['loads'].append(dict(source=str(source), actual=actual))
    load, overlay, initialize = XMLConfig.load, XMLConfig.overlay, MereologicalCodes.__init__
    def loaded(config, path):
        result = load(config, path)
        apply(config, path)
        return result
    def overlaid(config, path):
        result = overlay(config, path)
        apply(config, path)
        return result
    def initialized(index, *args, **kwargs):
        assert TheXMLConfig.space('ConceptualSpace', key) == value
        initialize(index, *args, **kwargs)
        row = dict(meaning_width=index.context_width, centroid=index.symbol_centroid,
            membership_priming=TheXMLConfig.space('ConceptualSpace', 'membershipPriming'))
        if part == 'B': assert row['meaning_width'] == 0
        if part == 'C': assert row['centroid'] is False
        if part == 'D': assert row['membership_priming'] is False
        log['instances'].append(row)
    apply(TheXMLConfig, 'existing singleton')
    try:
        with patch.object(XMLConfig, 'load', loaded), patch.object(XMLConfig, 'overlay', overlaid), \
                patch.object(MereologicalCodes, '__init__', initialized):
            yield
        assert log['instances'], 'no model index constructed under the control'
        log['verified'] = True
    finally:
        (folder / 'effective-switch.json').write_text(json.dumps(log, indent=2) + '\n')


def child(kind, part, folder):
    import rng_replay
    rng_replay.switches = lambda: effective_switch(part, folder)
    import campaign
    if kind == 'sum':
        campaign.sum_child(folder)
        return
    import pytest
    raise SystemExit(pytest.main(['-q', *campaign.XOR]))


def preflight():
    from util import XMLConfig, TheXMLConfig
    # Verify public setters after both loading and overlaying, without models
    # or training. Runtime construction has separate assertions in each child.
    for key, value in [('meaningWidth', 0), ('symbolCentroid', False), ('membershipPriming', False)]:
        config = XMLConfig(ROOT / 'data/XOR_grammar.xml', ROOT / 'data/model.xml')
        config.set('ConceptualSpace.' + key, value)
        assert config.space('ConceptualSpace', key) == value


def main():
    import bounded_tests as bounded
    import campaign
    preflight()
    OUT.mkdir(exist_ok=False)
    snapshot = bounded.source_snapshot(ROOT)
    assert snapshot == json.loads((HERE / 'delivered-source/source.json').read_text())
    original = json.loads((HERE / 'bisections/plan.json').read_text())
    jobs = list(original['jobs'])
    bounded.write_json(OUT / 'plan.json', dict(jobs=jobs,
        source='Same original failed standing entries, with effective XML settings asserted',
        seed=None, standing_retries=0, standing_replacements=0,
        prior_diagnostics='bisections/: invalid, constructor patch never reached live singleton; all remained on',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    active, done = [], []
    while jobs or active:
        assert snapshot == bounded.source_snapshot(ROOT)
        for job in active[:]:
            result = job['process'].poll()
            if result is not None:
                bounded.write_json(job['folder'] / 'process.json', result)
                done.append({k:v for k,v in job.items() if k not in ('folder','process')} | dict(process=result))
                active.remove(job)
                print(json.dumps(done[-1]), flush=True)
        while jobs and len(active) < 3:
            job = jobs.pop(0)
            folder = OUT / job['name']
            folder.mkdir()
            env = campaign.environment()
            env.update(OPERATORS_REPLAY_RNG=str(HERE / 'measurements' / f'{job["kind"]}-{job["run"]:02}' / 'unseeded-entry.pt'),
                       OPERATORS_DISABLE=job['part'])
            if job['kind'] == 'xor':
                env.update(PYTEST_PLUGINS='operators_gate_observer', ITEM7_XOR_GATE='5',
                    ITEM7_XOR_MEASUREMENTS=str(folder / 'observations.jsonl'),
                    REVIEW17_REPORTS=str(folder / 'reports.jsonl'))
            command = [sys.executable, str(Path(__file__).resolve()), job['kind'], job['part'], str(folder)]
            process = bounded.GuardedProcess(command, cwd=ROOT, env=env, log_path=folder / 'run.log',
                memory_bytes=8*bounded.GIB, timeout=1800).start()
            active.append(job | dict(folder=folder, process=process))
        bounded.write_json(OUT / 'progress.json', dict(done=done, pending=len(jobs), active=[j['name'] for j in active]))
        time.sleep(.5)
    bounded.write_json(OUT / 'complete.json', dict(jobs=done, source_matched=snapshot == bounded.source_snapshot(ROOT)))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        child(sys.argv[1], sys.argv[2], Path(sys.argv[3]))
    else:
        main()
