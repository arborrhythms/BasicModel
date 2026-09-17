"""The quick suite skips expensive checks before setup; RUN_SLOW=1 opts in."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


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
