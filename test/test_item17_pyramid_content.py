"""The reconstruction's content read from the bytes (round 6d of item 1.7,
seventeenth pass, item 1; §19.135, 4): at a place, the `.what` of the byte
there and of each form of the percept store present in the bytes there
(`FieldReverse.content`, the forms those present in the input's bytes,
`FieldReverse.forms_in`), on both sides of the pyramid match alike.
Alec's assert, outside compiled code: the content of a byte string, taken
back to bytes, is that byte string; and the error is nothing exactly when
the bytes put out are the input's (the input itself, one byte changed, two
words exchanged, the same bytes cut into other forms). Then, in the model:
a wrong word in its place against the right word 1, 2 and 4 places over,
with the bytes' content and with this one; and the time of one call and of
the charge in a training step. The model's start is drawn afresh and
printed; nothing asserted of what a start gives."""
import os
import time

import torch
from torch.nn import functional as F

from Layers import Error


def store(model):
    from FieldReverse import byte_rows
    radix = model.perceptualSpace.percept_store
    rows = byte_rows(radix)
    codes = (radix._basis.lookup_rows(rows.clamp_min(0)) * (rows >= 0)[:, None]).detach().float()
    return radix, rows, codes


def placed(text, width, at=0):
    """Bytes ``[1, width]`` with ``text`` at ``at`` and zero elsewhere, and
    the places it fills."""
    raw = torch.zeros(1, width, dtype=torch.long)
    raw[0, at:at + len(text)] = torch.tensor(list(text.encode()))
    valid = torch.zeros(1, width, dtype=torch.bool)
    valid[0, at:at + len(text)] = True
    return raw, valid


def pasted(pieces, width):
    """Graded bytes ``[1, width, 256]``: each piece ``(text, at)`` pasted as
    one-hot bytes at its place, overlapping pieces adding."""
    amounts = torch.zeros(1, width, 256)
    for text, at in pieces:
        for i, byte in enumerate(text.encode()):
            if 0 <= at + i < width:
                amounts[0, at + i, byte] += 1.
    return amounts


class Content:
    """The model's content of graded bytes, with the forms present in one
    input's bytes."""
    def __init__(self, model, wanted, width, forms=True):
        from FieldReverse import forms_in
        self.radix, self.rows, self.codes = store(model)
        raw, valid = placed(wanted, width)
        self.form_rows, self.spellings = forms_in(self.radix, raw, valid) if forms else (
            torch.zeros(0, dtype=torch.long), torch.zeros(0, 0, dtype=torch.long))
        self.form_codes = self.radix._basis.lookup_rows(self.form_rows).detach().float()
        self.wanted = self(pasted([(wanted, 0)], width))

    def __call__(self, amounts):
        from FieldReverse import content
        return content(amounts, self.codes, self.spellings, self.form_codes)

    def error(self, amounts):
        K, total = Error.pyramid_match(self(amounts), self.wanted)
        return total - K, total


def back_to_bytes(content, target, admitted, width):
    """The bytes whose content is ``target``, by search: place by place, the
    admitted byte (or none) whose string's content is nearest, swept until
    nothing changes."""
    chosen = [None] * width
    def amounts():
        out = torch.zeros(1, width, 256)
        for p, byte in enumerate(chosen):
            if byte is not None:
                out[0, p, byte] = 1.
        return out
    for _ in range(6):
        before = list(chosen)
        for p in range(width):
            best, best_cost = chosen[p], None
            for byte in [None] + admitted:
                chosen[p] = byte
                cost = float((content(amounts()) - target).abs().sum())
                if best_cost is None or cost < best_cost:
                    best, best_cost = byte, cost
            chosen[p] = best
        if chosen == before:
            break
    return bytes(b for b in chosen if b is not None)


def lesson_model(tmp_path, monkeypatch):
    from test_item17_places import minimal, model_for, nominate
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    model = model_for(tmp_path, monkeypatch)
    one, _, lesson = minimal()
    nominate(model, one + lesson[:8])
    return model


def test_the_content_is_a_function_of_the_bytes_and_inverts_to_them(tmp_path, monkeypatch):
    model = lesson_model(tmp_path, monkeypatch)
    try:
        sentence, width = 'This is Felix.', 16
        content = Content(model, sentence, width)
        forms = [model.perceptualSpace.percept_store.bytes_for(int(r)) for r in content.form_rows]
        print('forms present in the input:', forms)
        assert forms, 'the lesson model holds the forms of its words'
        admitted = [v for v in range(256) if int(content.rows[v]) >= 0]
        # (a) The content of a byte string, taken back to bytes, is that byte string.
        for text in (sentence, 'Here is Ciao.', 'is This Felix?'):
            target = content(pasted([(text, 0)], width))
            assert back_to_bytes(content, target, admitted, width) == text.encode()
        # (b) The error is nothing exactly when the bytes put out are the input's.
        cases = {'the input itself': [(sentence, 0)],
                 'the same bytes cut into other forms': [('Th', 0), ('is is Fe', 2), ('lix.', 10)],
                 'one byte changed': [('This is Felix?', 0)],
                 'two words exchanged': [('Felix is This.', 0)]}
        for name, pieces in cases.items():
            error, total = content.error(pasted(pieces, width))
            print(f'{name}: error {float(error):.6g} of {float(total):.6g}')
            if name in ('the input itself', 'the same bytes cut into other forms'):
                assert float(error) <= 1e-6 * float(total)
            else:
                assert float(error) > 1e-3 * float(total)
        # The graded presence of a form takes a gradient.
        amounts = (pasted([(sentence, 0)], width) * .9).requires_grad_(True)
        error, _ = content.error(amounts)
        error.backward()
        assert torch.isfinite(amounts.grad).all() and amounts.grad.abs().sum() > 0
    finally:
        model.End()


def test_measured_a_wrong_word_in_its_place_and_the_right_word_moved(tmp_path, monkeypatch):
    model = lesson_model(tmp_path, monkeypatch)
    try:
        sentence, width = 'This is Felix.', 32
        for forms in (False, True):
            content = Content(model, sentence, width, forms=forms)
            ratio = lambda pieces: (lambda e, t: round(float(e / t), 4))(*content.error(pasted(pieces, width)))
            wrong = ratio([('Here is Felix.', 0)])
            moved = {k: ratio([('This', k), (' is Felix.', 4)]) for k in (1, 2, 4)}
            print(f'{"forms and bytes" if forms else "bytes alone"}: "Here" for "This" in its place {wrong}'
                  f' | "This" moved by k (k: error) {moved}')
        # One call at the lesson's size (64 places), beside the bytes alone.
        width = 64
        raw = pasted([(sentence, 0)], width)
        amounts = torch.softmax(torch.randn(1, width, 256) * 4, -1)
        for forms in (False, True):
            content = Content(model, sentence, width, forms=forms)
            def call():
                from FieldReverse import forms_in
                if forms:
                    forms_in(content.radix, *placed(sentence, width))
                Error.pyramid_match(content(amounts), content(raw))
            call()
            t = time.perf_counter()
            for _ in range(50):
                call()
            print(f'one call at 64 places, {"forms and bytes" if forms else "bytes alone"}: '
                  f'{(time.perf_counter() - t) / 50 * 1e3:.3f} ms')
    finally:
        model.End()


def test_measured_the_charge_in_a_training_step(tmp_path, monkeypatch):
    import SentenceField
    from test_item18_expect import read
    model = lesson_model(tmp_path, monkeypatch)
    charge, spent = SentenceField.charge, []

    pyramid = Error.pyramid
    def checked_pyramid(errors, name, put, wanted, **options):
        assert not wanted.requires_grad
        return pyramid(errors, name, put, wanted, **options)

    monkeypatch.setattr(Error, 'pyramid', checked_pyramid)

    def timed(*args, **kw):
        t = time.perf_counter()
        result = charge(*args, **kw)
        spent.append(time.perf_counter() - t)
        return result

    monkeypatch.setattr(SentenceField, 'charge', timed)
    try:
        optimizer = model.getOptimizer(lr=.01)
        steps = []
        for text in ('This is Felix.', 'Here is Ciao.', 'What is that?') * 2:
            t = time.perf_counter()
            read(model, text, training=True, optimizer=optimizer)
            model.End()
            steps.append(time.perf_counter() - t)
        print(f'training step {sum(steps[2:]) / len(steps[2:]) * 1e3:.1f} ms; the charge in it '
              f'{sum(spent[2:]) / max(1, len(spent[2:])) * 1e3:.2f} ms ({len(spent)} charges)')
        assert spent
    finally:
        model.End()
