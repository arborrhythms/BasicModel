"""Failing probe: a closing with no selected individual must not mint one."""
import torch
from test_independent_components import component_fixture
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning


def test_kind_only_closing_does_not_mint_an_individual():
    columns = component_fixture()
    existing = columns.nouns.ids
    store = TernaryTruthStore(4, capacity=8)
    columns.owner._closed_clause_store = lambda: store
    columns._allocate = lambda direction: 100
    meaning = ConceptualMeaning.from_description(torch.eye(4)[2])
    for position in range(4):
        row = store.append_meaning(meaning, document_key='extension',
            sentence_index=position, order=2, kind='observation')
        columns.commit(row, meaning)
    assert columns.nouns.ids == existing
