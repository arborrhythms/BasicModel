"""Native percept synthesis keeps its membership cube and location band."""
from types import SimpleNamespace

import torch
import pytest

from Spaces import PartSpace


def test_part_synthesis_needs_no_fold_layer():
    events = torch.tensor([[[.2, .9, -.4, .3], [.8, .1, .4, .3]],
                           [[.2, .9, -.4, .3], [.8, .1, .4, .3]]],
                          requires_grad=True)
    host = SimpleNamespace(nDim=2, _radix_part_events=lambda ids, offsets: events)
    result = PartSpace.synthesize_word_parts(
        host, torch.tensor([[0, 1], [0, 1]]),
        torch.tensor([[True, True], [False, False]]))
    torch.testing.assert_close(result, torch.tensor([[[.8, .9, -.4, .3]],
                                                    [[0., 0., 0., 0.]]]))
    result[..., :2].sum().backward()
    torch.testing.assert_close(events.grad[0, :, :2], torch.tensor([[0., 1.], [1., 0.]]))
    assert events.grad[1].count_nonzero() == 0


def test_native_towers_have_no_learned_fold_layers(tmp_path):
    from test_grounded_xor import grounded_model
    model, _ = grounded_model(tmp_path)
    for space, names in ((model.perceptualSpace, ('sigma', 'sigmas', '_sigma_stack_modules')),
                         (model.wholeSpace, ('pi', 'pis', '_pi_stack_modules'))):
        assert not any(hasattr(space, name) for name in names)


def test_old_subsymbolic_noop_spelling_is_rejected(tmp_path):
    from util import XMLConfig
    path = tmp_path / 'retired.xml'
    path.write_text('<model><architecture><subsymbolicNoop>1</subsymbolicNoop></architecture></model>')
    with pytest.raises(ValueError, match='subsymbolicNoop'):
        XMLConfig().load(str(path))
