"""The dictionary consumer's identity check is independent of native decoding."""
from types import SimpleNamespace
import torch
from operators_run_audit import identity_recovered


def test_dictionary_identity_recovery_detects_a_changed_bit_with_clean_native_store():
    native = torch.tensor([[1., 0., .3], [0., 1., .7]])
    identity = SimpleNamespace(word_rows={b'a': 0, b'b': 1},
                               audit=lambda: {'reconstruction_errors': 0})
    model = SimpleNamespace(perceptualSpace=SimpleNamespace(
        percept_store=SimpleNamespace(identity=identity),
        subspace=SimpleNamespace(what=SimpleNamespace(lookup_rows=lambda row: native[row]))))
    owner = SimpleNamespace(similarity_codebook=SimpleNamespace(
        mereology=SimpleNamespace(identity_width=2)),
        word_surface_for_row=lambda row: (b'a', b'b')[row])
    assert identity_recovered(model, owner, [0, 1], native)
    assert identity_recovered(model, owner, [], native[:0])
    dictionary = native.clone()
    dictionary[0, 0] = 0.
    assert identity.audit()['reconstruction_errors'] == 0
    assert not identity_recovered(model, owner, [0, 1], dictionary)
    dictionary = native.clone()
    dictionary[:, 2] = 1.
    assert identity_recovered(model, owner, [0, 1], dictionary)
