"""All lexical readers consume the definition index, without former caches."""
from types import SimpleNamespace
import torch
from test_item7_definitions import model, admit


def test_generation_uses_the_definition_vocabulary_without_a_row_spelling_cache():
    m = model()
    cs = m._concept_owner()
    _, obj = admit(m, 'hello')
    cs.__dict__.pop('_row_surfaces', None)
    cs.__dict__.pop('_surface_object_rows', None)
    point = cs.similarity_codebook.getW()[cs._csw_row_of(obj)].detach()
    assert m._generated_word_text(point[None, None], torch.tensor([1])) == ('hello',)


def test_reference_update_law_uses_the_definition_index():
    from Spaces import Space, reference_update_mask
    from test_codebook_update_law import _knob
    m = model()
    cs = m._concept_owner()
    _, obj = admit(m, 'hello')
    vq = SimpleNamespace()
    space = Space.__new__(Space)
    object.__setattr__(space, 'subspace', SimpleNamespace(codebook=lambda: SimpleNamespace(vq=vq)))
    space.serial_mode = False
    _knob('on')
    try:
        assert space.install_reference_update_law(lambda: cs.definitions, side='object')
        assert torch.equal(vq.update_mask_fn(32, None), reference_update_mask(False, [obj], 32))
    finally:
        _knob(None)
