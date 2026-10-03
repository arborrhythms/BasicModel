"""The quick suite skips expensive checks before setup; RUN_SLOW=1 opts in."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_weekly_includes_the_parked_inline_suites():
    import slow_weekly
    assert slow_weekly.INLINE_SUITES == (
        'bin/Legacy.py', 'bin/etc/SPNN.py', 'bin/etc/SigmaPi.py', 'bin/etc/SymPercept.py')


def test_weekly_source_copy_survives_edits_to_the_working_tree(tmp_path):
    import slow_weekly
    source = tmp_path / 'working'
    (source / 'bin').mkdir(parents=True)
    (source / 'bin/module.py').write_text('value = 1\n')
    (source / 'data').mkdir()
    (source / 'data/model.xml').write_text('<model/>\n')
    destination = tmp_path / 'measurement-source'
    slow_weekly.snapshot_source(source, destination)
    (source / 'bin/module.py').write_text('value = 2\n')
    assert (destination / 'bin/module.py').read_text() == 'value = 1\n'
    assert (destination / 'data/model.xml').read_text() == '<model/>\n'


@pytest.mark.parametrize('setting,expected', [(None, '1 passed, 1 skipped'),
                                               ('0', '1 passed, 1 skipped'),
                                               ('1', '2 passed')])
def test_slow_mode_controls_collection_before_expensive_fixture_setup(tmp_path, setting, expected):
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    (tmp_path / 'test_example.py').write_text(
        'import pytest\nfrom pathlib import Path\n'
        '@pytest.fixture\ndef expensive():\n'
        " Path('expensive-ran').write_text('yes')\n"
        'def test_quick(): pass\n'
        '@pytest.mark.slow\ndef test_training(expensive): pass\n')
    env=dict(os.environ, PYTHONPATH=str(Path(__file__).parent))
    env.pop('RUN_SLOW', None)
    if setting is not None:
        env['RUN_SLOW']=setting
    result=subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', 'slow_tests', '-p', 'no:cacheprovider',
         '-q', '--confcutdir', str(tmp_path), str(tmp_path / 'test_example.py')],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert expected in result.stdout
    assert (tmp_path / 'expensive-ran').exists() is (setting == '1')


def test_slow_selection_includes_historical_and_mps_switches(tmp_path):
    """Weekly selection cannot silently omit old decorators or MPS opt-ins."""
    (tmp_path / 'pytest.ini').write_text('[pytest]\n')
    (tmp_path / 'test_switches.py').write_text(
        'import os,pytest,unittest\n'
        '_RUN_SLOW = os.getenv("RUN_SLOW") == "1"\n'
        '@pytest.mark.skipif(not _RUN_SLOW, reason="old opt-in")\n'
        'def test_old_pytest(): pass\n'
        '@unittest.skipIf(not _RUN_SLOW, "old unittest opt-in")\n'
        'class TestOld(unittest.TestCase):\n'
        ' def test_class(self): pass\n'
        '@pytest.mark.skipif(os.getenv("RUN_MPS_SLOW") != "1", reason="MPS opt-in")\n'
        '@pytest.mark.parametrize("width", [16,64])\n'
        'def test_mps(width): pass\n'
        'def test_ordinary(): pass\n')
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent),
               RUN_SLOW='1', RUN_MPS_SLOW='1')
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', '-p', 'slow_tests', '-p', 'no:cacheprovider',
         '-q', '-m', 'slow', '--confcutdir', str(tmp_path), str(tmp_path / 'test_switches.py')],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '4 passed, 1 deselected' in result.stdout
