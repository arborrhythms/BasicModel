"""Default mechanism certificates for the lesson-artifact prerequisite."""
import hashlib
import json
from pathlib import Path

import pytest

from phase_artifacts import PhaseArtifact, PhaseRegistry


def test_registry_round_trip(tmp_path):
    registry = PhaseRegistry({'pending': None, 'taught': PhaseArtifact(
        'trained.ckpt', 'ab'*32, 'receipt.md', config='model.xml', minimum_sentences=123)})
    path = tmp_path/'registry.json'
    registry.write(path)
    restored = PhaseRegistry.read(path)
    assert restored.phases == registry.phases
    assert json.loads(path.read_text())['phases']['pending'] is None
    from bounded_tests import source_snapshot
    fixtures=tmp_path/'test/fixtures'
    fixtures.mkdir(parents=True)
    frozen=fixtures/'phase-artifacts.json'
    registry.write(frozen)
    before=source_snapshot(tmp_path)
    registry.phases['pending']=registry.phases['taught']
    registry.write(frozen)
    assert source_snapshot(tmp_path)!=before  # A sweep cannot change prerequisites mid-run.


def test_absent_phase_skips_wrong_hash_fails_and_restoration_is_required(tmp_path, monkeypatch):
    from bounded_tests import run_suite
    artifact, receipt = tmp_path/'trained.ckpt', tmp_path/'receipt.md'
    artifact.write_bytes(b'completed lesson state')
    receipt.write_text('Producer receipt for this fixture.\n')
    good = PhaseArtifact(str(artifact), hashlib.sha256(artifact.read_bytes()).hexdigest(), str(receipt))
    registry = PhaseRegistry(dict(absent=None, missing=PhaseArtifact(
        str(tmp_path/'absent.ckpt'),good.sha256,str(receipt)),
        wrong=PhaseArtifact(str(artifact),'0'*64,str(receipt)), present=good))
    path = tmp_path/'registry.json'
    registry.write(path)
    monkeypatch.setenv('BASICMODEL_PHASE_ARTIFACTS',str(path))
    monkeypatch.delenv('RUN_SLOW',raising=False)
    (tmp_path/'conftest.py').write_text('pytest_plugins = ["slow_tests", "phase_artifacts"]\n')
    (tmp_path/'test_phase.py').write_text('''import pytest
from pathlib import Path
@pytest.mark.phase('absent')
def test_absent(restore_phase):
    raise AssertionError('absent artifacts must skip before the body')
@pytest.mark.phase('missing')
def test_missing(restore_phase):
    raise AssertionError('missing artifact files must skip before the body')
@pytest.mark.phase('wrong')
def test_wrong(restore_phase):
    raise AssertionError('bad hashes must fail before restoration')
@pytest.mark.slow
@pytest.mark.phase('present')
def test_present(restore_phase):
    assert restore_phase(Path.read_bytes) == b'completed lesson state'
@pytest.mark.phase('present')
def test_rejected(restore_phase):
    restore_phase(lambda path: False)
''')
    result = run_suite(root=tmp_path, selectors=['test_phase.py'],run_dir=tmp_path/'result',
        memory_bytes=512*1024**2,timeout=30,suite_timeout=90,lock_path=tmp_path/'lock')
    assert result['exit_code']==1 and result['completed']==result['selected']
    reports={row['nodeid'].split('::')[-1]:row for worker in result['workers'] for row in worker['reports']}
    assert reports['test_present']['outcome']=='passed'  # PHASE_CERTIFICATES outranks slow.
    for name in ('test_absent','test_missing'):
        assert reports[name]['outcome']=='skipped' and 'phase artifact absent' in reports[name]['message']
    assert reports['test_wrong']['outcome']=='failed' and 'hash mismatch' in reports['test_wrong']['message']
    assert reports['test_rejected']['outcome']=='failed' and 'could not be restored' in reports['test_rejected']['message']
    assert set(result['skipped_certificates_by_phase'])=={'absent','missing'}
    assert len(result['phase_certificates'])==5
    manifest=json.loads((tmp_path/'result/source-manifest.json').read_text())
    assert manifest['phase_registry_override']==dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    html=(tmp_path/'result/report.html').read_text()
    assert 'Skipped phase certificates' in html and 'absent (1)' in html and 'missing (1)' in html
