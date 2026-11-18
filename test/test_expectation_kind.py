"""Item 4 retires the slot/kind head; grammatical kind remains observed metadata."""
import copy
import torch
from Layers import SymbolExpectation


def test_grammatical_kind_does_not_train_a_second_expectation_head():
    from test_sentence_expectation import layer
    model = layer()
    assert not any('kind' in name for name, _ in model.named_parameters())
