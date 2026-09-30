import json
import torch
from Models import BasicModel
from test_reasoning_cde_model import TestReasoningCDEModel


def test_normal_policy_metadata_diagnostic(monkeypatch):
    original = BasicModel._program_entries
    def observe(self, *args, **kwargs):
        try:
            return original(self, *args, **kwargs)
        except IndexError:
            values = {name: list(value.shape) for name, value in vars(self.symbolSpace).items()
                      if name.startswith('_word_reference') and torch.is_tensor(value)}
            values['program_positions'] = list(args[0][0].shape)
            print(json.dumps(values))
            raise
    monkeypatch.setattr(BasicModel, '_program_entries', observe)
    TestReasoningCDEModel.setUpClass()
    TestReasoningCDEModel().test_training_step_uses_the_normal_policy_configuration()
