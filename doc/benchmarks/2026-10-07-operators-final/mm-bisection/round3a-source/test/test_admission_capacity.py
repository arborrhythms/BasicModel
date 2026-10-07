import pytest

import torch

from Layers import RadixLayer

from Spaces import Codebook, Embedding


def test_admission_at_capacity_is_atomic_and_names_the_physical_knob():
    store = RadixLayer(4, initial_cap=2)
    store.insert(b'a')
    store.insert(b'b')
    parameter = store._basis.W
    before = parameter.detach().clone()
    with pytest.raises(RuntimeError, match='nVectors'):
        store.insert(b'c')
    assert len(store) == 2 and store.get_id(b'c') is None
    assert store._basis.W is parameter
    torch.testing.assert_close(parameter, before)


def test_atomic_byte_admission_preflights_the_complete_batch():
    store = RadixLayer(4, initial_cap=2)
    store.insert(b'a')
    parameter = store._basis.W
    with pytest.raises(RuntimeError, match='nVectors'):
        store.ensure_atomic_bytes([b'b', b'c'])
    assert len(store) == 1
    assert store.get_id(b'b') is None and store.get_id(b'c') is None
    assert store._basis.W is parameter


def test_admission_preserves_optimizer_parameter_and_moments():
    # Standalone radix fixtures default to a nonlearned tensor; production
    # PartSpace owns a Parameter. Match that ownership before checking Adam.
    basis = Codebook()
    basis.create(1, 4, 4, customVQ=False)
    basis.setW(torch.nn.Parameter(basis.W.detach().clone()))
    store = RadixLayer(4, initial_cap=4, basis=basis)
    store.insert(b'a')
    parameter = store._basis.W
    optimizer = torch.optim.Adam([parameter], lr=.01)
    store.active_prototypes().sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    moment = optimizer.state[parameter]['exp_avg'].clone()
    store.ensure_atomic_bytes([b'b', b'c'], optimizer=optimizer)
    assert store._basis.W is parameter
    assert optimizer.param_groups[0]['params'] == [parameter]
    torch.testing.assert_close(optimizer.state[parameter]['exp_avg'], moment)
    assert tuple(parameter.shape) == (4, 4)


def test_checkpoint_cannot_resize_an_existing_percept_codebook():
    source = RadixLayer(4, initial_cap=4)
    source.insert(b'a')
    target = RadixLayer(4, initial_cap=2)
    parameter = target._basis.W
    with pytest.raises(ValueError, match='nVectors'):
        target.load_vocab_extras(source.vocab_extras())
    assert target._basis.W is parameter and len(target) == 0


def test_lexicon_reserves_its_physical_capacity_before_admission():
    lexicon = Embedding()
    lexicon.create(nInput=4, nVectors=256, nDim=4)
    parameter = lexicon.wv._vectors
    assert tuple(parameter.shape) == (256, 4)
    lexicon.insert('new-word')
    lexicon.insert('another-word')
    assert lexicon.wv._vectors is parameter
    assert tuple(parameter.shape) == (256, 4)
    assert lexicon.pretrain.optimizer.param_groups[0]['params'] == [parameter]

