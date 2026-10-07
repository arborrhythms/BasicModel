"""B's history must refer only to the current retained LTM population."""
import torch
from types import SimpleNamespace
import Spaces
from Layers import TernaryTruthStore
from Meaning import ConceptualMeaning
from test_independent_components import component_fixture


def test_unretained_innovation_cannot_change_population_cost(monkeypatch):
    columns = component_fixture()
    store = TernaryTruthStore(4, capacity=8)
    meaning = ConceptualMeaning.from_description(torch.eye(4)[0])
    store.append_meaning(meaning, document_key='retained', sentence_index=0, kind='observation')
    owner = columns.owner
    owner._closed_clause_store = lambda: store
    owner._order0_inventory_row = lambda row: row < 4
    owner.similarity_codebook = SimpleNamespace(W=torch.eye(4), lookup_rows=lambda rows: torch.eye(4)[rows])
    monkeypatch.setattr(Spaces, '_concept_alloc_of', lambda _: SimpleNamespace(
        layer=lambda: SimpleNamespace(_tensor_row_keys={i: i for i in range(4)})))
    prediction = SimpleNamespace(roles=torch.zeros(3, 4))
    columns.begin_forward()
    expected = columns.cost([meaning], [prediction], torch.tensor([True]))
    absent = max(store.row_ids[:len(store)].tolist()) + 100
    columns._innovations[absent] = torch.ones(3 * columns.capacity)
    columns.begin_forward()
    actual = columns.cost([meaning], [prediction], torch.tensor([True]))
    torch.testing.assert_close(actual, expected)
