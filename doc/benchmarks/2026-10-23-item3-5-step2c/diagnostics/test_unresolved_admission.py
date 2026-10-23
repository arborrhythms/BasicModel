"""An unavailable typed interpretation cannot change the kept reading's shape."""
from types import SimpleNamespace
import torch
from Language import LanguageSpace


def test_unresolved_typed_admission_keeps_the_scored_source_reference():
    rule = SimpleNamespace(method_name='kind', head_role=1, determiner_mode='kind',
        reference_roles=('argin1',), reference_modes=(), order_delta=0)
    language = SimpleNamespace(_compose_binary_rules=(), _compose_unary_rules=(rule,))
    owner = SimpleNamespace(resolve_word_concept=lambda *a, **k: None,
        word_surface_for_row=lambda row: 'pending-form', word_concepts=lambda surface: (101,),
        _concept_source_order=lambda identity: 0,
        definitions=SimpleNamespace(word=lambda **kwargs: 101),
        interpret=SimpleNamespace(forward=lambda *a, **k: None))
    arguments = (owner, torch.tensor([10]), torch.tensor([101]),
                 torch.tensor([[0, -1, 0], [2, 0, 0]]))
    results = [LanguageSpace.resolve_lexical_references(language, *arguments,
        leaf_orders=torch.tensor([0]), admit=admit) for admit in (False, True)]
    assert results[0][0].tolist() == results[1][0].tolist() == [101]
    assert results[0][1].tolist() == results[1][1].tolist() == [2]


def test_ambiguous_typed_reading_is_open_in_preview_and_admission():
    rule = SimpleNamespace(method_name='kind', head_role=1, determiner_mode='kind',
        reference_roles=('argin1',), reference_modes=(), order_delta=0)
    language = SimpleNamespace(_compose_binary_rules=(), _compose_unary_rules=(rule,))
    owner = SimpleNamespace(resolve_word_concept=lambda *a, **k: None,
        word_surface_for_row=lambda row: 'pending-form', word_concepts=lambda surface: (201, 202),
        _concept_source_order=lambda identity: 2,
        definitions=SimpleNamespace(word=lambda **kwargs: 101),
        interpret=SimpleNamespace(forward=lambda *a, **k: None))
    arguments = (owner, torch.tensor([10]), torch.tensor([101]),
                 torch.tensor([[0, -1, 0], [2, 0, 0]]))
    results = [LanguageSpace.resolve_lexical_references(language, *arguments,
        leaf_orders=torch.tensor([0]), admit=admit) for admit in (False, True)]
    assert results[0][0].tolist() == results[1][0].tolist() == [-1]
    assert results[0][1].tolist() == results[1][1].tolist() == [2]
