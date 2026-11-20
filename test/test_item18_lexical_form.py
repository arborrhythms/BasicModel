"""Higher-order symbols read form from their word without storing it."""
import pytest
import torch
from nomination_fixtures import recurrent_interpretation

pytestmark = pytest.mark.assessment_rung(1)


def named_kind(model, form):
    owner = model._concept_owner()
    word, *_ = recurrent_interpretation(owner, 
        model.perceptualSpace.percept_store.spell_out(form.encode()),
        model.wholeSpace.property_rows_for_bytes(form), key=form)
    from definition_fixtures import agreeing_views
    agreeing_views(owner, word)
    kind = owner.new_concept(order=1)
    owner.interpret.define(word, kind)
    return word, kind


@pytest.mark.parametrize('configuration', ('aligned', 'mixing'))
def test_91_reading_and_free_generation_take_the_mapped_words_form(
        tmp_path, monkeypatch, eager_reading, configuration):
    from Generative import lexical_candidates
    from PrependedSymbols import word_view
    if configuration == 'aligned':
        from test_compiled_word_chunk import _tiny_canonical_model
        model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
        inputs = ['hello']
    else:
        from recon_bench import _build_model
        model, *_ = _build_model('data/XOR_grammar.xml')
        inputs, _ = model.inputSpace.getTestData()
    model._tensor_peer_while_eager = True
    owner = model._concept_owner()
    word, kind = named_kind(model, 'hello')
    row = owner._concept_row_of(kind)
    width = owner.similarity_codebook.mereology.percept_event_width
    expected = owner.similarity_codebook.lookup_rows(owner._concept_row_of(word))[:width].detach()
    assert expected.any()
    assert not owner.similarity_codebook.lookup_rows(row)[:width].any()
    try:
        root = owner.similarity_codebook.lookup_rows(row)[None]
        bank, _ = lexical_candidates(owner, root, limit=16)
        assert (bank.rows == row).any()
        torch.testing.assert_close(bank.codes[bank.rows == row][..., :width],
            expected.expand(int((bank.rows == row).sum()), -1), rtol=0, atol=0)
        with torch.no_grad():
            model.forward(model.inputSpace.prepInput(inputs))
        source = model.inputSpace
        if configuration == 'aligned':
            rows = word_view(source, 'object_rows')
            atoms = source._ar_word_object_atoms
        else:
            rows, atoms = source._ar_grammar_object_rows, source._ar_grammar_object_atoms
        assert (rows == row).any()
        torch.testing.assert_close(atoms[rows == row][..., :width],
            expected.expand(int((rows == row).sum()), -1), rtol=0, atol=0)
        assert not owner.similarity_codebook.lookup_rows(row)[:width].any()
    finally:
        if configuration == 'aligned':
            from assessment_support import record
            record(model, dict(word='hello', words=['hello']))
        model.End()


def test_91_spelling_presence_reads_form_and_keeps_its_gradient(
        tmp_path, monkeypatch, eager_reading):
    from test_compiled_word_chunk import _tiny_canonical_model
    from SentenceUnderstanding import PrimedSymbols
    from OutputOccurrences import generated, realize
    from WordSurfaces import stage
    model = _tiny_canonical_model(tmp_path, monkeypatch, batch_size=1)
    model._tensor_peer_while_eager = True
    owner = model._concept_owner()
    word, kind = named_kind(model, 'wug')
    row = owner._concept_row_of(kind)
    width = owner.similarity_codebook.mereology.percept_event_width
    code = owner.similarity_codebook.lookup_rows(row).detach().clone()
    code[:width] = owner.similarity_codebook.lookup_rows(owner._concept_row_of(word))[:width]
    # This terminal's word is present, while its claim has neither lane.
    code[width:] = 0.
    leaf = code[None, None].clone().requires_grad_()
    byte, valid, present, support = stage([[b'wug']], owner.word_surface_capacity,
        device=code.device, eos=True)
    bank = PrimedSymbols(torch.tensor([[row]], device=code.device), code[None,None],
        code.new_ones(1,1), torch.zeros(1,1,dtype=torch.bool,device=code.device),
        byte, valid, meaning_start=width, surface_present=present, primitive_support=support)
    stop = len(model.languageSpace._generate_binary_ops)+len(model.languageSpace._generate_unary_ops)
    try:
        result = generated(model, leaf, torch.tensor([1]), torch.tensor([False]),
            bank, (torch.tensor([[stop]]),), depths=torch.tensor([1]))
        strength = result.realization_field.evidence.sum()
        torch.testing.assert_close(strength, code.new_tensor(1.), rtol=0, atol=0)
        gradient, = torch.autograd.grad(strength, leaf)
        assert gradient[..., :width].abs().sum() > 0
        assert not gradient[..., width:].any()
        with torch.no_grad():
            assert result.texts(realize(model, result).materialize()) == ('wug',)
        assert not owner.similarity_codebook.lookup_rows(row)[:width].any()
    finally:
        from assessment_support import record
        record(model, dict(word='wug', words=['wug']))
        model.End()
