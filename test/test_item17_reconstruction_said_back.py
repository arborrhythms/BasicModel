"""Measure sentence reconstruction from opening evidence before grammar or thought.
The teaching and measurement use the same model and its ordinary sentence path.
"""
import os

import torch

from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, minimal, model_for, nominate, words_of

WORDS = words_of(LESSON) + COUNTING[:6]


class Held:
    """Observe the input surface at opening and the rules chosen afterwards."""
    def __init__(self, model, monkeypatch):
        import SentenceField
        self.rows, self.rules = [], []
        language = model.languageSpace
        inverse, choose = SentenceField.reconstruction, language.choose_operation

        def reconstructed(owner, field):
            reverse = inverse(owner, field)
            surface = bytes(reverse.materialize()[0, 0].tolist()).rstrip(b'\0')
            self.rows.append(dict(joined=surface.decode('latin1')))
            return reverse

        def chosen(state, row_gate, **kwargs):
            result = choose(state, row_gate, **kwargs)
            choice = result[0] if isinstance(result, tuple) else result
            if bool(choice.applied[0]) and bool(row_gate.reshape(-1)[0]):
                rules = {1: language._compose_binary_rules, 2: language._compose_unary_rules}
                names = rules.get(int(choice.kind[0]), ())
                index = int(choice.local_op[0])
                if 0 <= index < len(names):
                    self.rules.append(names[index].method_name)
            return result

        monkeypatch.setattr(SentenceField, 'reconstruction', reconstructed)
        monkeypatch.setattr(language, 'choose_operation', chosen)


def measured(model, held, sentences):
    from test_item18_expect import read
    reconstructed = 0
    for sentence in sentences:
        held.rows.clear()
        held.rules.clear()
        with torch.no_grad():
            read(model, sentence)
            model.End()
        actual = held.rows[0]['joined']
        reconstructed += actual == sentence
        print(f'{sentence!r} reconstructed as {actual!r}, compose rules {held.rules}')
    return dict(reconstructed=reconstructed)


def test_the_row_at_one_by_the_models_reconstruction(tmp_path, monkeypatch):
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    model = model_for(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson + (' '.join(COUNTING[:4]) + '.', ' '.join(COUNTING[4:]) + '.'))
        held = Held(model, monkeypatch)
        alone = tuple(w + '.' for w in WORDS)
        optimizer = model.getOptimizer()
        # Taught at the model's own rate (fourth part of the nineteenth
        # pass), on what is measured: the words alone, the two-word
        # sentences and the lesson's sentences.
        taught = alone + two + tuple(dict.fromkeys(lesson))
        rounds = int(os.environ.get('ROUNDS', '10'))
        for r in sorted({0, rounds}):
            if r:
                for _ in range(rounds):
                    for text in taught:
                        read(model, text, training=True, optimizer=optimizer)
                        model.End()
            print('taught', r, 'rounds of the 19 words alone, the two-word sentences and the lesson\'s sentences')
            words = measured(model, held, alone)
            print(' row at 1.00, of', len(alone), 'words alone:', words)
            sentences = measured(model, held, ('This is Felix.',) + LESSON[1:4])
            print(' lesson sentences, of 4:', sentences)
            pairs = measured(model, held, two)
            print(' row at 1.75, of', len(two), 'two-word sentences:', pairs)
        assert words['reconstructed'] <= len(alone)
    finally:
        model.End()


def text(t):
    return bytes(t.detach().long().flatten().tolist()).rstrip(b'\0').decode('latin1')


def test_unit_the_perceptual_inverse_paste_and_the_pyramid(tmp_path, monkeypatch):
    """A unit test of the lower stretch's layers (`FieldReverse.reverse` and
    the error `Layers.Error.pyramid`), not the model saying a word back. On
    "This is Felix.", read once with a gradient: (1) the gradient of the
    model's own reconstruction error reaching the order-zero Sigma (the rows
    that read the `.what`), the first Pi's `.where` entries (the rows that
    read the `.where`) and the grid (in no optimizer); (2) the paste itself,
    with a word's wires moved (the pyramid's own measures are in
    test_pyramid_match.py)."""
    import SentenceField
    from dataclasses import replace
    from FieldReverse import reverse, byte_rows
    from Layers import Error
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    model = model_for(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson)
        owner = model._concept_owner()
        sigma, basic, registry = owner.install_order_zero(), owner.basic_percepts, model.where_registry
        held = {}
        charge = SentenceField.charge
        def charged(model, errors, name, carrier, field, active, **kw):
            wanted = charge(model, errors, name, carrier, field, active, **kw)
            held.setdefault('field', field)
            held.setdefault('wanted', wanted.detach())
            value = errors._value(errors._terms[name + '.pyramid'])
            with torch.enable_grad():
                grads = ([None] * 3 if not value.requires_grad else
                         torch.autograd.grad(value.sum(), [sigma.value, basic.where_entries, registry.grid],
                                             allow_unused=True, retain_graph=True))
            print('in the read, gradient of', name + '.pyramid', round(float(value.sum()), 4),
                  {k: None if g is None else float(g.norm()) for k, g in zip(('sigma', 'where_entries', 'grid'), grads)})
            return wanted
        monkeypatch.setattr(SentenceField, 'charge', charged)
        read(model, 'This is Felix.', training=True, optimizer=model.getOptimizer(lr=0.))
        model.End()
        monkeypatch.setattr(SentenceField, 'charge', charge)
        field, wanted = held['field'], held['wanted']
        print('grid in an optimizer:', any(p is registry.grid for g in model.getOptimizer().param_groups
                                            for p in g['params']))
        # The paste: one word's wires moved.
        radix = model.perceptualSpace.percept_store
        sub = model.conceptualSpace.subspace
        live = (field.where[0, :, 1] > field.where[0, :, 0]).nonzero().flatten().tolist()
        def moved_wires(f, k, by):
            wires = f.wires.clone()
            wires[0, k] = registry.encode('text', torch.tensor(int(f.where[0, k, 0]) + by))
            return replace(f, wires=wires)
        for name, f in (('as read', field), ('"Felix" wires moved by 4', moved_wires(field, live[2], 4))):
            with torch.no_grad():
                out = reverse(model, sub, f)
                rows = byte_rows(radix)
                put = out._radix_activity.amounts @ (radix._basis.lookup_rows(rows.clamp_min(0))
                                                     * (rows >= 0)[:, None]).to(wanted)
                K, total = Error.pyramid_match(put, wanted)
                print(f'paste, {name}: {text(radix.decode_activity(out._radix_activity)[0])!r}, pyramid error',
                      round(float(1 - K[0] / total[0]), 4))
        assert held
    finally:
        model.End()


def test_open_reconstruction_is_independent_of_generate_rules(tmp_path, monkeypatch):
    """The input cost reaches the opening inverse before any generate rule executes."""
    import SentenceField
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)
    model = model_for(tmp_path, monkeypatch)
    use_eager_reading(monkeypatch)
    language = model.languageSpace
    rules = [(rule.method_name, op) for rule, op in zip(
        (*language._generate_binary_rules, *language._generate_unary_rules),
        (*language._generate_binary_ops, *language._generate_unary_ops))]
    reached, charge = {}, SentenceField.charge

    def charged(model_, errors, name, carrier, field, active, **kw):
        wanted = charge(model_, errors, name, carrier, field, active, **kw)
        value = errors._value(errors._terms[name + '.pyramid'])
        for rule, op in rules:
            params = [p for p in (op.parameters() if isinstance(op, torch.nn.Module) else ()) if p.requires_grad]
            if not params or not value.requires_grad:
                continue
            with torch.enable_grad():
                grads = torch.autograd.grad(value.sum(), params, allow_unused=True, retain_graph=True)
            norm = sum(float(g.norm()) for g in grads if g is not None)
            entry = reached.setdefault(rule, dict(reads=0, reached=0))
            entry['reads'] += 1
            entry['reached'] += norm > 0
        return wanted

    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson)
        monkeypatch.setattr(SentenceField, 'charge', charged)
        optimizer = model.getOptimizer(lr=0.)
        for sentence in ('This.', 'Felix.', 'This is Felix.', 'Here is Ciao.'):
            read(model, sentence, training=True, optimizer=optimizer)
            model.End()
        assert reached and all(value['reached'] == 0 for value in reached.values()), reached
    finally:
        model.End()


def test_reconstruction_extends_only_with_agreeing_proper_occurrences():
    from types import SimpleNamespace
    from FieldReverse import assemble_parts, codebook, spelling

    # The name alone spells a different sequence. The inverse can only get
    # hello by following proper witnesses. At frontier 3 the larger overlap
    # wins even over an earlier-listed competing extension ("lxx").
    name = b'xxxxx'
    children = ((name, 0, 5), (b'hel', 0, 3), (b'lxx', 2, 5),
                (b'llo', 2, 5), (b'ello', 1, 5))
    identity = SimpleNamespace(nested={name: children}, revision=0)
    basic = SimpleNamespace(index=[0], native=SimpleNamespace(identity=identity), revision=0,
                            value_codes=torch.zeros(1), form_of=lambda _: name,
                            place_rows={}, conjunctions={}, is_value=lambda _: False)
    rows, table = codebook(basic)
    assert rows == [0] and bytes(table[0].tolist()) == spelling(basic, 0) == b'hello'
    for parts, expected in (
            (((name, 0, 5),), b''),
            (((b'hel', 0, 3), (b'llo', 2, 5)), b'hello'),
            (((b'hel', 0, 3), (b'bad', 2, 5)), b'hel'),
            (((b'ha', 0, 2), (b'ha', 2, 4)), b'haha')):
        identity.nested[name] = parts
        identity.revision += 1
        assert spelling(basic, 0) == expected
        projected = codebook(basic)[1][0]
        assert bytes(projected[projected >= 0].tolist()) == expected
    parts = torch.tensor([[[104, 101, 108], [108, 108, 111]]])
    actual = torch.compile(assemble_parts, fullgraph=True, backend='eager')(parts, torch.tensor([[0, 2]]), 5)
    assert bytes(actual[0].tolist()) == b'hello'
