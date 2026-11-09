"""The generate menu reads the same formal concepts as compose."""
from types import SimpleNamespace
import torch
from PartsOfSpeech import formal_arguments
from GenerateOccupation import features
from argument_fixtures import argument_inventory
from test_parts_of_speech import rule
from test_item20_round34_projection import record


def test_generate_features_equal_argument_scores_and_kept_output_is_preferred():
    binary = rule('intersection', head=0)
    owner, speech, words = argument_inventory(formal_arguments([binary]))
    from IdentityRows import point
    values = torch.stack([point(owner, word) for word in words])
    program = SimpleNamespace(leaves=values[:2], word_ids=torch.tensor(words[:2]),
        actions=torch.tensor([[0,-1,0],[0,-1,1],[1,0,0]]))
    speech.observe_kept(program, [binary], [])
    from ConceptHierarchy import refresh
    refresh(owner)
    values = torch.stack([point(owner, word) for word in words])
    speech.prepare_values()
    table = speech.argument_scores(torch.tensor(words), values)
    torch.testing.assert_close(speech.value_scores(values), table)
    language = SimpleNamespace(parts_of_speech=speech, _generate_binary_rules=(binary,), _generate_unary_rules=())
    parents = values[[0,2]]
    left = values[[0,0]][:,None].expand(-1,2,-1)
    right = values[[1,1]][:,None].expand(-1,2,-1)
    result = features(language, parents, left, right)
    expected = torch.stack((table[[0,2],speech.index['intersection_argout1']],
        table[[0,0],speech.index['intersection_argin1']], table[[1,1],speech.index['intersection_argin2']]),-1)
    torch.testing.assert_close(result[:,0], expected)
    assert result[:,1].count_nonzero() == 0
    assert result[0,0,0] > result[1,0,0]
    record('output-typing', dict(features=result.tolist(), argument_scores=expected.tolist(),
        occupied=float(result[0,0,0]), unoccupied=float(result[1,0,0]), stop_zero=True))


def test_undeclared_fixed_operation_contributes_zero_scores():
    owner, speech, words = argument_inventory(('intersection_argin1','intersection_argout1'))
    from IdentityRows import point
    values = point(owner, words[0])[None]
    fixed = rule('part')
    language = SimpleNamespace(parts_of_speech=speech, _generate_binary_rules=(fixed,), _generate_unary_rules=())
    assert features(language, values, values[:,None].expand(-1,2,-1), values[:,None].expand(-1,2,-1)).count_nonzero() == 0
