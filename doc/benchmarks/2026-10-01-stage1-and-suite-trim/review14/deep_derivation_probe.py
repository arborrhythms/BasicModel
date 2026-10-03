"""Small reproduction of LM_5M's unbounded Python recursion in trace export."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import traceback
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT/'bin'))
import torch
from Models import BasicModel


def owner(words=600):
    slots = 3*words+4
    ids = torch.full((1, slots), -1, dtype=torch.long)
    ids[:, 3:3*words:3] = 0
    ids[:, 2:3*words:3] = 1
    trace = SimpleNamespace(choices=lambda: (ids, None, ids >= 0),
                           _choice_positions=torch.zeros_like(ids))
    language = SimpleNamespace(_cs_binary_rule_ids=torch.tensor([0]),
        _cs_unary_rule_ids=torch.tensor([1]),
        local_op_from_rule_ids=lambda values, catalog:
            (torch.zeros_like(values), values == catalog[0]))
    return SimpleNamespace(_reconstruction_stack=lambda: trace,
        languageSpace=language, inputSpace=SimpleNamespace(
            _word_active_mask=torch.ones(1, words, dtype=torch.bool)),
        conceptualSpace=SimpleNamespace(stm=SimpleNamespace(capacity=2)))


def check():
    words = 600
    positions, actions, targets, columns = BasicModel._derivation_program(
        owner(words), budget=3*words)
    assert positions.tolist() == [list(range(words))]
    expected = [(0, -1, 0), (2, 0, -1)]
    addresses = [-1, 2]
    for word in range(1, words):
        expected.extend(((0, -1, word), (1, 0, -1), (2, 0, -1)))
        addresses.extend((-1, 3*word, 3*word+2))
    assert actions.tolist() == [[list(action) for action in expected]]
    assert columns.tolist() == [addresses]
    assert targets[0, :6].tolist() == [1, 0, 2, 1, 0, 2]
    assert targets[0, -3:].tolist() == [1, 2, -1]


if __name__ == '__main__':
    try:
        check()
    except Exception as exc:
        Path(sys.argv[1]).write_text(json.dumps(dict(error=repr(exc), traceback=traceback.format_exc()), indent=2)+'\n')
        raise
    Path(sys.argv[1]).write_text(json.dumps(dict(passed=True, words=600, actions=1799), indent=2)+'\n')
