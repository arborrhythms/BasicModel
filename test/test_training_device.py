"""Training test workers must honor an accelerator and keep the quick default."""
import pytest

import bounded_tests as runner


@pytest.mark.parametrize('device', ['gpu', 'mps', 'cuda:0'])
def test_explicit_training_device_is_preserved(tmp_path, monkeypatch, device):
    monkeypatch.setenv('BASICMODEL_DEVICE', device)
    assert runner.worker_environment(tmp_path)['BASICMODEL_DEVICE'] == device


def test_slow_training_defaults_to_a_required_accelerator(tmp_path, monkeypatch):
    monkeypatch.delenv('BASICMODEL_DEVICE', raising=False)
    monkeypatch.setenv('RUN_SLOW', '1')
    assert runner.worker_environment(tmp_path)['BASICMODEL_DEVICE'] == 'gpu'


def test_quick_suite_retains_its_cpu_default(tmp_path, monkeypatch):
    monkeypatch.delenv('BASICMODEL_DEVICE', raising=False)
    monkeypatch.setenv('RUN_SLOW', '0')
    assert runner.worker_environment(tmp_path)['BASICMODEL_DEVICE'] == 'cpu'


def test_all_dispatches_slow_training_to_gpu_and_ordinary_checks_to_cpu(tmp_path, monkeypatch):
    import torch
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip('real accelerator required for the mixed-device integration')
    monkeypatch.delenv('BASICMODEL_DEVICE', raising=False)
    monkeypatch.setenv('RUN_SLOW', '1')
    monkeypatch.delenv('PYTEST_PLUGINS', raising=False)
    (tmp_path / 'pytest.ini').write_text('[pytest]\nmarkers =\n    slow: substantial training\n')
    (tmp_path / 'test_small.py').write_text(
        "import os\nfrom pathlib import Path\n"
        "def check(name):\n"
        " assert os.environ['BASICMODEL_DEVICE'] == 'cpu'\n"
        " with Path('devices').open('a') as f: f.write(name+':cpu\\n')\n"
        "def test_before(): check('before')\n"
        "def test_after(): check('after')\n")
    (tmp_path / 'test_training.py').write_text(
        "import os,pytest\nfrom pathlib import Path\n"
        "@pytest.mark.slow\ndef test_training():\n"
        " import torch\n"
        " device=os.environ['BASICMODEL_DEVICE']\n"
        " assert device == 'mps' or device.startswith('cuda')\n"
        " x=torch.ones(4,device=device,requires_grad=True)\n"
        " x.square().sum().backward()\n"
        " assert x.grad.device.type == torch.device(device).type\n"
        " with Path('devices').open('a') as f: f.write('training:'+device+'\\n')\n")
    selected=['test_small.py::test_before', 'test_training.py::test_training',
              'test_small.py::test_after']
    result=runner.run_suite(root=tmp_path, selectors=selected, run_dir=tmp_path/'result',
        memory_bytes=1024**3, timeout=60, suite_timeout=240,
        batch_size=64, lock_path=tmp_path/'lock')
    assert result['exit_code'] == 0, result['reason']
    assert result['selected'] == result['completed'] == selected
    devices=(tmp_path/'devices').read_text().splitlines()
    assert devices[0]=='before:cpu' and devices[-1]=='after:cpu'
    assert devices[1] in ('training:mps','training:cuda')
    assert result['collection']['device']=='cpu'
    assert [worker['device'] for worker in result['workers']] == [
        'cpu', devices[1].split(':',1)[1], 'cpu']
