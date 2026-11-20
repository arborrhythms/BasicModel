"""What a few known words read after a short lesson: each word concept's
two lanes at its own reading in the open run. Prints; asserts nothing.
Run: pytest test/membership_gradient_probe.py -s"""
import torch


def test_probe(tmp_path, monkeypatch):
    from test_item17_looks import nominated_lesson
    from test_item18_expect import read
    texts = ('alpha gamma', 'beta delta', 'alpha delta', 'beta gamma')
    model = nominated_lesson(tmp_path, monkeypatch, texts)
    try:
        optimizer = model.getOptimizer(lr=.01)
        for presentation in range(24):
            read(model, texts[presentation % 4], training=True, optimizer=optimizer)
        owner = model._concept_owner()
        for text in texts:
            read(model, text)
            field = model._sentence_open_fields[0]
            ids = field.concept_ids if field.concept_ids.ndim == 1 else field.concept_ids[:, 0]
            values = {}
            for position, word in enumerate(text.split()):
                cid = owner.interpret.word_for_form(word)
                at = (ids == cid).nonzero().flatten()
                values[word] = [round(float(v), 4) for v in field.evidence[at[0], 0, position]] if len(at) else None
            print('KNOWN', text, values)
    finally:
        model.End()
