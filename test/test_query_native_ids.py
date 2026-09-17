"""A held grammatical program retains native referents, independently of rows."""
import pytest
import torch

from Understanding import AnswerProgram
from test_output_path_supervised import _native_answer_model
from test_output_walk import _capture_program_probe


def test_real_sentence_capture_owns_native_object_ids_across_later_staging(tmp_path):
    model = _native_answer_model(tmp_path, False)
    model.eval()
    try:
        with torch.no_grad():
            held = _capture_program_probe(model, ['1 plus 2', '3 plus 4'])
            source = model.inputSpace
            expected = torch.where(source._ar_word_object_rows >= 0,
                                   source._ar_word_object_ids,
                                   source._ar_word_concept_ids)
            active = source._word_active_mask
            assert bool((expected[active] > 0).any())
            snapshots = []
            for row, program in enumerate(held.answer_program):
                torch.testing.assert_close(program.concept_ids, expected[row, active[row]])
                # A concept ID is an allocator address, not its dictionary row.
                known = program.concept_ids > 0
                assert bool((program.concept_ids[known] != program.rows[known]).any())
                snapshots.append(program.concept_ids.clone())
            _capture_program_probe(model, ['5 plus 6', '7 plus 8'])
        for program, snapshot in zip(held.answer_program, snapshots):
            torch.testing.assert_close(program.concept_ids, snapshot)
            durable = program.detached()
            torch.testing.assert_close(durable.concept_ids, snapshot.cpu())
            assert durable.concept_ids.data_ptr() != program.concept_ids.data_ptr()
    finally:
        model.End()
        model.symbolSpace.soft_reset()
        torch._dynamo.reset()


def test_legacy_program_rows_are_never_reinterpreted_as_native_concept_ids():
    program = AnswerProgram(
        rows=torch.tensor([7, 3]), word_rows=torch.tensor([4, 6]),
        activations=torch.ones(2), leaves=torch.ones(2, 8),
        actions=torch.empty(0, 3, dtype=torch.long), targets=torch.tensor([0, 1]),
        end_state=torch.ones(3, 8))
    assert program.concept_ids.tolist() == [-1, -1]


def test_missing_object_identity_cannot_fall_back_to_a_different_word_referent():
    from types import SimpleNamespace
    from Models import BasicModel
    source = SimpleNamespace(
        _ar_word_concept_rows=torch.tensor([[3, 4, 5]]),
        _ar_word_object_rows=torch.tensor([[11, -1, 13]]),
        _ar_word_concept_ids=torch.tensor([[101, 102, 103]]),
        _ar_word_object_ids=torch.tensor([[201, -1, -1]]))
    model = SimpleNamespace(inputSpace=source)
    ids = BasicModel._word_symbol_concept_ids(model)
    assert ids.tolist() == [[201, 102, -1]]


def test_fractional_staged_identity_is_not_silently_truncated_into_a_referent():
    from types import SimpleNamespace
    from Models import BasicModel
    source = SimpleNamespace(
        _ar_word_concept_rows=torch.tensor([[3]]),
        _ar_word_object_rows=torch.tensor([[11]]),
        _ar_word_concept_ids=torch.tensor([[101]]),
        _ar_word_object_ids=torch.tensor([[201.9]]))
    with pytest.raises(TypeError, match='integer|identity|IDs'):
        BasicModel._word_symbol_concept_ids(SimpleNamespace(inputSpace=source))
