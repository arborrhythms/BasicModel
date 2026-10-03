"""Step 1: the capability gate must measure the four emitted answers."""
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize('predictions, passes', [
    ([0., 1., 1., 0.], True),
    ([.49, .51, .51, .49], False),
    ([0., 1., .49, 0.], False),
])
def test_gate_measures_answers_not_placeholder(monkeypatch, predictions, passes):
    import test_explicit_dimensions as gate
    data = SimpleNamespace(
        reconstructed_output=[torch.tensor([v]) for v in predictions],
        test_output=[torch.tensor([v]) for v in (0., 1., 1., 0.)])
    model = SimpleNamespace(rCorrect=torch.zeros(1),
        inputSpace=SimpleNamespace(data=data))
    monkeypatch.setattr(gate, '_run_xor_grammar_in_process', lambda: model)
    case = gate.TestXorGrammarLearnsXor()
    if passes:
        case.test_xor_class_accuracy()
    else:
        with pytest.raises(AssertionError):
            case.test_xor_class_accuracy()
