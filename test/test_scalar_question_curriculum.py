"""The acquisition criterion measures native structure and has a fixed bound."""
from types import SimpleNamespace as NS
import pytest
import torch
from scalar_question_curriculum import bounded_stage, CurriculumBoundReached, reading_forest


def test_curriculum_requires_three_consecutive_rounds_with_all_three_readings():
    target = ['surface', 0, 1]
    good = [[target]] * 3
    wrong = [[target], [], [target]]
    rounds = iter([good, good, wrong, good, good, good])
    trained, reports = [], []
    result = bounded_stage(lambda: trained.append(1), lambda: next(rounds), target,
                           report=reports.append)
    assert result['round'] == 6
    assert len(trained) == 6
    assert [row['consecutive'] for row in reports] == [1, 2, 0, 1, 2, 3]


def test_curriculum_stops_at_200_rounds_and_keeps_the_failure_observations():
    trained, reads = [], []
    def train():
        trained.extend((0, 1, 2))
        return [{'probability': .25}]
    def read():
        reads.extend((0, 1, 2))
        return [[], [], []]
    with pytest.raises(CurriculumBoundReached) as error:
        bounded_stage(train, read, ['lower', 0])
    assert len(trained) == len(reads) == 600
    assert error.value.report['round'] == 200
    assert error.value.report['readings'] == [[], [], []]
    assert error.value.report['taught_probabilities'] == [{'probability': .25}]


def test_reading_measurement_distinguishes_open_bind_and_native_source_places():
    language = NS(_compose_binary_rules=[NS(method_name='bind'), NS(method_name='surface')],
                  _compose_unary_rules=[NS(method_name='lower_unary', surface_name='lower')])
    actions = torch.tensor([[0, -1, 0], [0, -1, 1], [2, 0, 0], [1, 0, -1], [1, 1, -1]])
    program = NS(actions=actions, operation_refs=torch.zeros(5, 2, dtype=torch.long))
    expected = ['surface', 0, ['bind.open', ['lower', 1]]]
    assert reading_forest(language, program, [0, 1]) == [expected]
    assert reading_forest(language, program, [1, 0]) != [expected]
    program.operation_refs[3, 1] = 123
    assert reading_forest(language, program, [0, 1]) == [['surface', 0, ['bind', ['lower', 1]]]]
