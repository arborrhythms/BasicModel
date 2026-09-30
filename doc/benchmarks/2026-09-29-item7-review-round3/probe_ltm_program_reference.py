"""Diagnose the LTM-only clause reference path without changing its reader."""
import json
from pathlib import Path

import pytest
import torch

from ClauseRow import Clause
from Meaning import ConceptualMeaning
from Understanding import AnswerProgram
from test_item7_definitions import admit
from test_mm_xor import _fresh_model

HERE = Path(__file__).resolve().parent


def test_ltm_clause_reference_reaches_inventory_only_program_reader():
    model = _fresh_model('data/MM_grammar_wording.xml')[0]
    try:
        cs, store = model._concept_owner(), model.symbolSpace.ltm_store
        from MemoryIndex import configure_model_index
        configure_model_index(model, store)
        word, obj = admit(model, 'hello')
        object_row = cs._csw_row_of(obj)
        point = cs.similarity_codebook.getW()[object_row].detach().clone()
        roles = torch.stack((point, point * 0, point * 0))
        clause = Clause(ConceptualMeaning(roles, torch.tensor([True, False, False])),
                        point=point, refs=(obj, -1, -1), subject_word_id=word, order=1)
        row = store.write_clause(clause)
        row_id = int(store.row_ids[row])
        assert cs.resolve_word_concept('hello', order=1) == row_id
        assert cs._csw_row_of(row_id) is None
        torch.testing.assert_close(store.point_of_row(row_id), point)
        program = AnswerProgram(rows=torch.tensor([object_row]),
            word_rows=torch.tensor([object_row]), activations=torch.ones(1),
            leaves=point[None], actions=torch.tensor([[0, -1, 0]]),
            targets=torch.tensor([0]), end_state=roles,
            concept_ids=torch.tensor([obj]), reference_ids=torch.tensor([row_id]),
            reference_orders=torch.tensor([1]))
        with pytest.raises(ValueError, match='no allocated payload row') as raised:
            model.languageSpace.program_meaning(program, model.grammatical_thoughts)
        (HERE / 'ltm-program-reference-diagnosis.json').write_text(json.dumps(dict(
            diagnostic_only=True, configuration='data/MM_grammar_wording.xml', seed=None,
            word_id=word, object_id=obj, clause_row_id=row_id,
            clause_point_available_in_ltm=True, clause_has_inventory_row=False,
            lexical_order_lookup_returns_clause=True,
            program_reader_error=str(raised.value),
            note='The diagnostic confirms the reader gap; it does not repair or pass the native packed-reading probe.'
        ), indent=2) + '\n')
    finally:
        model.End()
        model.symbolSpace.soft_reset()
