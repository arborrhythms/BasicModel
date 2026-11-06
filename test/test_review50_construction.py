"""§50 items106–107: explicit candidate shapes and replayable corpus randomness."""
from dataclasses import replace
import random

import pytest
import torch

from SentenceUnderstanding import PrimedSymbols


@pytest.mark.parametrize('field', ('codes', 'weights', 'own', 'bytes', 'byte_valid',
                                   'forms', 'surface_present', 'primitive_support'))
def test_primed_table_must_belong_to_its_candidate_rows(field):
    bank = PrimedSymbols(torch.zeros(2, 3, dtype=torch.int64), torch.zeros(2, 3, 8),
        torch.ones(2, 3), torch.ones(2, 3, dtype=torch.bool),
        torch.zeros(2, 3, 64, dtype=torch.uint8), torch.zeros(2, 3, 64, dtype=torch.bool),
        forms=torch.zeros(2, 3, 4), surface_present=torch.zeros(2, 3, dtype=torch.bool),
        primitive_support=torch.zeros(2, 3, 256, dtype=torch.bool))
    wrong = getattr(bank, field)[:, :1]
    with pytest.raises(ValueError, match=f'PrimedSymbols.{field}.*differs from rows'):
        replace(bank, **{field: wrong})
    # This independent structural table is rebuilt after the lexical join.
    assert replace(bank, form_wholes=object()).form_wholes is not None


def test_unselected_math_generator_is_owned_by_recorded_python_stream():
    from exact import MathProblemGenerator
    initial = random.getstate()
    try:
        first = MathProblemGenerator(seed=None)
        after = random.getstate()
        expected = [first.rng.getrandbits(64) for _ in range(8)]
        random.setstate(initial)
        second = MathProblemGenerator(seed=None)
        assert random.getstate() == after
        assert [second.rng.getrandbits(64) for _ in range(8)] == expected
    finally:
        random.setstate(initial)
