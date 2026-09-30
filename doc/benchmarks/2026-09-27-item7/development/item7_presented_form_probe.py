from dataclasses import replace
import torch
from ClauseSeal import clause_from_program
from test_item7_acceptance import SentenceFixture


def test_selected_anaphor_form_binds_to_its_own_sealed_particular(monkeypatch):
    f = SentenceFixture(monkeypatch)
    prior = f.store.seal_clause(f.clause(('lift', 'cat', 'runs')))
    entry = f.program(('lift', 'he', 'rests'))
    refs = entry.concept_ids.clone()
    refs[0] = f.store.row_ids[prior]
    values = entry.leaves.clone()
    values[0] = f.store.point_of_row(int(refs[0]))
    entry = replace(entry, reference_ids=refs, reference_orders=torch.tensor([1, -1]),
                    reference_values=values)
    clause = clause_from_program(f.language, entry, registry=f.registry)
    row = f.store.seal_clause(clause)
    assert int(f.store.row_ids[row]) in f.cs.word_concepts('he')
    assert int(f.store.row_ids[row]) not in f.cs.word_concepts('rests')
    assert f.cs.interpret.forward(f.words['he'][0], order=1) == f.noun('he')
