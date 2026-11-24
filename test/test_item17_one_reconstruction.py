"""One reconstruction (round 6d of item 1.7, sixteenth pass, item 5;
§19.134; seventeenth pass, item 4, §19.135, 3): in a model that reads (no
symbols, or a grammar of no rule), the one
reconstruction's upper stretch is empty and it is the perceptual inverse of
the basic-level concepts alone. These are tests of that function
(`SentenceField.reconstruction`) against the shorter loop it replaced (the
reverse of the open field, `model.reverse(..., field=open_field)`, and for
the numeric path `NumericPercepts.reverse`): the same content put back, and
the same error. On `MM_xor` (no symbols, numeric percepts) and on a text
model whose grammar has no rule."""
import os

import torch


def test_mm_xor_one_reconstruction_is_the_numeric_inverse_alone():
    from test_mm_xor import _fresh_model
    from SentenceField import reconstruction
    m, _, data = _fresh_model()
    try:
        m.eval()
        loader = data.data_loader(split='train', num_streams=4)
        inputs, _ = next(iter(loader))
        with torch.no_grad():
            understanding = m.understand(m.inputSpace.prepInput(inputs))
        field = understanding.reconstruction_carriers.get('field')
        assert field is not None and m.numeric_percepts is not None
        with torch.no_grad():
            put = reconstruction(m, field)
            shorter = m.numeric_percepts.reverse(field)
        torch.testing.assert_close(put.materialize().reshape(shorter.shape), shorter, rtol=0, atol=0)
    finally:
        m.End()


def _grammar_free_text_model(tmp_path, monkeypatch):
    """The small text model of the looks' tests with a grammar of no rule."""
    from test_compiled_word_chunk import _tiny_canonical_model
    import util
    grammar = tmp_path / 'none.grammar'
    grammar.write_text('<?xml version="1.0"?>\n<grammar name="none">\n  <compose>\n  </compose>\n</grammar>\n')
    monkeypatch.setattr(util, 'TheCompileBackend', 'none')
    model = _tiny_canonical_model(tmp_path, monkeypatch, word_buckets='8', input_width=32, concept_rows=64,
                                  training_overrides={'reconstructionPlacement': 'eager'}, grammar=grammar)
    model._tensor_peer_while_eager = True
    model.inputSpace.data.has_supervised_outputs = False
    return model


def test_text_model_with_no_grammar_one_reconstruction_is_the_open_fields_inverse(tmp_path, monkeypatch):
    import SentenceField
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    model = _grammar_free_text_model(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    held = []
    one = SentenceField.reconstruction
    def observed(model_, field, **kw):
        result = one(model_, field, **kw)
        with torch.no_grad():
            shorter = model_.reverse(model_.conceptualSpace.subspace.carrier_like(), field=field)
        held.append(dict(record=kw.get('record'), programs=kw.get('programs'), result=result, shorter=shorter))
        return result
    monkeypatch.setattr(SentenceField, 'reconstruction', observed)
    try:
        print('scope', model._understanding_reconstruction_scope())
        from test_item17_places import nominate
        nominate(model, ('This is Felix.', 'Here.'))
        held.clear()
        for text in ('This is Felix.', 'Here.'):
            read(model, text, training=True, optimizer=model.getOptimizer(lr=0.))
            model.End()
        assert held
        for row in held:
            put = row['result']
            shorter = row['shorter']
            # A model that reads (a grammar of no rule) puts back the
            # basic level from itself: no walk (§19.135, 3).
            assert torch.equal(put.materialize(), shorter.materialize())
            torch.testing.assert_close(put._radix_activity.amounts, shorter._radix_activity.amounts, rtol=0, atol=0)
            torch.testing.assert_close(put._reconstruction_percepts, shorter._reconstruction_percepts,
                                       rtol=0, atol=0)
        print('calls', len(held), 'with a record', sum(row['record'] is not None for row in held),
              'put back', [bytes(int(v) for v in row['result'].materialize().reshape(-1).tolist()
                                 if 0 < int(v) < 256) for row in held])
    finally:
        model.End()
