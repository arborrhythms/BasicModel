"""Declared graded presentation around the unchanged original training driver.

Preparation changes only presentation order, as §14.7.8 requests. The native
driver, corpus, verifier and observer remain the original imported sources.
The reporting wrapper observes greedy openings and per-epoch episode work.
This helper cannot start until protocol.json exists and its certificate passes.
"""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = HERE.parent/'2026-10-07-math-chain'
sys.path[:0] = [str(ROOT/'bin'), str(ROOT/'test'), str(OLD)]
spec = importlib.util.spec_from_file_location('original_math_train', OLD/'math_train.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.HERE, driver.ROOT = HERE, ROOT


def declaration():
    protocol = json.loads((HERE/'protocol.json').read_text())
    result = json.loads((HERE/protocol['development_certificate_folder']/'result.json').read_text())
    assert result['status'] == 'completed' and result['completed_epochs'] == 1
    acceptance = json.loads((HERE/protocol['development_acceptance']) .read_text())
    assert acceptance['passed'] and acceptance['criterion'] == 'work_per_declarative_episode_falls'
    import hashlib
    assert acceptance['result_sha256'] == hashlib.sha256(
        (HERE/protocol['development_certificate_folder']/'result.json').read_bytes()).hexdigest()
    assert protocol['attention_budget'] == 32 and protocol['ltm_capacity'] == 131072
    assert protocol['runs_per_condition'] == 10 and protocol['seed'] is None and protocol['retries'] == 0
    assert protocol['presentation_order'] == 'graded_by_chain_length'
    assert protocol['epochs'] == int(28800/result['epoch_seconds'])
    return protocol


def prepare(folder, epochs):
    protocol = declaration()
    assert epochs == protocol['epochs']
    driver.prepare(folder, epochs)
    path = folder/'presentations.json'
    presentations = json.loads(path.read_text())
    for presentation in presentations:
        presentation['train'].sort(key=lambda doc: -1 if doc['pair'] is None else doc['pair'][1])
    path.write_text(json.dumps(presentations)+'\n')
    (folder/'presentation-order.json').write_text(json.dumps(dict(
        order=protocol['presentation_order'], comparison=None,
        premise_order='Original independently shuffled premises retained.'),indent=2)+'\n')


def train(folder, paired, condition, run):
    protocol = declaration()
    assert condition in protocol['conditions'] and 1 <= run <= protocol['runs_per_condition']
    from measurement_observer import Measurements
    with Measurements(folder):
        driver.train(folder, paired, condition, run)


if __name__ == '__main__':
    if sys.argv[1] == 'prepare':
        prepare(Path(sys.argv[2]),int(sys.argv[3]))
    elif sys.argv[1] == 'train':
        train(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],int(sys.argv[5]))
    else:
        raise ValueError('expected prepare or train')
