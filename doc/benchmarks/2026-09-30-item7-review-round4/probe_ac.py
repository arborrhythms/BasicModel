"""AC: object lookup is by the definition table; keep every owned-answer assertion."""
import pytest
import torch
from What import What
from test_output_path_supervised import _native_answer_model, _NATIVE_CONCEPT_WIDTHS

@pytest.mark.parametrize("concept_width", _NATIVE_CONCEPT_WIDTHS)
def test_legacy_parity_memory_cannot_change_an_owned_conceptual_answer(tmp_path, concept_width):
    from What import LTMSlot
    from Layers import WhatInteractionMemory
    from test_output_walk import _capture_program_probe
    m = _native_answer_model(tmp_path, False, concept_width=concept_width)
    m.eval()
    m.what_thinking_iterations = 2
    memory = WhatInteractionMemory(batch=2, capacity=8)
    object.__setattr__(m.symbolSpace, "what_memory", memory)
    questions = (What.supervised(0), What.supervised(1))
    try:
        with torch.no_grad():
            u = _capture_program_probe(m, ["1 plus 2", "3 plus 4"])
            base = torch.stack([field.end_state for field in u.sentence_states])
            module = m._ltm_attention(concept_width, 136, device=base.device, dtype=base.dtype)
            module["value"].weight.fill_(0.003)
            module["out"].weight.copy_(torch.eye(concept_width) * 0.1)
            response = torch.linspace(0.1, 0.3, 136)
            for b in range(2):
                memory.append_what_slot(LTMSlot(input=torch.zeros(136), output=response), b=b)
            deltas = torch.stack([m._attend_ltm(base[b, 0], [response]) for b in range(2)])
            assert bool(deltas.abs().sum() > 0)
            d = m._resolve_answer(u, questions)
            assert m.ltm_attention is module
            assert d.conceptual_answer.shape == (2, 3, concept_width)
            torch.testing.assert_close(d.conceptual_answer[:, 0], base[:, 0])
            torch.testing.assert_close(d.conceptual_answer[:, 1:], base[:, 1:])
            idea = m._materialize_answer_idea(u, d, questions)[0]
            torch.testing.assert_close(idea, d.conceptual_answer)
            # A lexical referent resolves through the direct concept index;
            # the completed row has no remembered word list.
            owner = m._concept_owner()
            word_id = owner.definitions.word(form='plus')
            concept_id, = owner.definitions.objects(word_id)
            expected = owner.similarity_codebook.getW()[owner._csw_row_of(concept_id)]
            query = m._referent_representation(
                u, 0, ('1', 'plus', '2'), 'plus', base[:1], detach=False)
            torch.testing.assert_close(query[0, 0], expected)
            # Later state changes cannot revise the resolved thinking result.
            _capture_program_probe(m, ["5 plus 6", "7 plus 8"])
            memory.append_what_slot(LTMSlot(input=torch.ones(136),
                                           output=torch.full((136,), 8.0)), b=0)
            again = m._materialize_answer_idea(u, d, questions)[0]
            torch.testing.assert_close(again, idea, rtol=0, atol=0)
    finally:
        m.End()
        m.symbolSpace.soft_reset()
        torch._dynamo.reset()
