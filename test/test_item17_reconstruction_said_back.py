"""The rows at 1.00 and 1.75 of doc/ModelAge.md by the model's one
reconstruction (round 6d of item 1.7, sixteenth pass, item 6; §19.134), input
to output on the model as it runs: each word alone (the word and its mark),
the two-word sentences and the lesson's sentences are given as the input and
the reconstruction put back is read. Beside each, as diagnostics only: the
words the leaves say, the grammar's applied rules, and the count of leaves
recovered from the idea, read from the record of the way up (a witness) or
given by no inverse, as `DerivationReconstruction.reconstruct` traces them.
A record: a start drawn afresh and printed, nothing asserted of what a start
gives. Then a unit test of the perceptual inverse's paste
(`FieldReverse.reverse`) and its error."""
import os

import torch

from test_item17_own_reading import COUNTING
from test_item17_places import LESSON, minimal, model_for, nominate, words_of

WORDS = words_of(LESSON) + COUNTING[:6]


def text(t):
    return bytes(int(v) for v in t.detach().reshape(-1).tolist() if 0 < int(v) < 256).decode('latin1')


class Held:
    """Each reading's reconstruction (``joined``), the words its leaves say,
    the leaves' sources and the grammar's applied rules."""
    def __init__(self, model, monkeypatch):
        import SentenceField
        language = model.languageSpace
        reconstruct, choose = SentenceField.reconstruct, language.choose_operation
        self.rules, self.rows = [], []
        owner = model._concept_owner()

        def rule(kind, op):
            rules = {1: language._compose_binary_rules, 2: language._compose_unary_rules}.get(kind, ())
            return getattr(rules[op], 'method_name', type(rules[op]).__name__) if 0 <= op < len(rules) else '-'

        def chosen(state, row_gate, **kw):
            result = choose(state, row_gate, **kw)
            c = result[0] if isinstance(result, tuple) else result
            if bool(c.applied[0]) and bool(row_gate.reshape(-1)[0]):
                self.rules.append(rule(int(c.kind[0]), int(c.local_op[0])))
            return result

        def held(model, record, errors, active, **kw):
            result = reconstruct(model, record, errors, active, **kw)
            sid = int(record.sentence)
            generation = model._sentence_generation
            f = generation.realization_field
            said = []
            for w in range(int(generation.counts[0])):
                presence = f.evidence[:, 0, w, 0]
                if not bool(presence.gt(0).any()):
                    said.append(None)
                    continue
                cid = int(f.concept_ids[int(presence.argmax())])
                row = owner._concept_row_of(cid)
                raw = None if row is None else owner.word_surface_for_row(row)
                said.append(None if raw is None else raw.decode('latin1'))
            sources = [node[2] for node in model._sentence_undone[0] if node[0] == 'leaf']
            policy = model.__dict__.pop('_last_walk_policy', None) or dict(charged=0, unheld=0)
            self.rows.append(dict(said=said, count=int(generation.counts[0]), sources=sources, policy=policy,
                                  bare=bool(model._sentence_bare[0]),
                                  joined=text(model._sentence_order_zero_reverse.materialize()[0, 0])))
            return result

        monkeypatch.setattr(language, 'choose_operation', chosen)
        monkeypatch.setattr(SentenceField, 'reconstruct', held)


def measured(model, held, sentences):
    from test_item18_expect import read
    tally = {'reconstructed': 0, 'leaves said the words': 0, 'leaves from the idea': 0,
             'leaves from the record': 0, 'leaves from no inverse': 0, 'no rule applied': 0}
    for sentence in sentences:
        held.rules.clear(); held.rows.clear()
        with torch.no_grad():
            read(model, sentence)
            model.End()
        row = held.rows[0]
        words = sentence[:-1].split() + [sentence[-1]]
        upper = [w for w in row['said'] if w is not None] == words
        tally['reconstructed'] += row['joined'] == sentence
        tally['leaves said the words'] += upper
        tally['no rule applied'] += row['bare']
        tally['policy: steps charged'] = tally.get('policy: steps charged', 0) + row['policy']['charged']
        tally['policy: steps not on its menu'] = tally.get('policy: steps not on its menu', 0) + row['policy']['unheld']
        for source in row['sources']:
            tally['leaves from no inverse' if source == 'none' else f'leaves from the {source}'] += 1
        print(f'  {sentence!r}: reconstruction {row["joined"]!r} | leaves {row["count"]} said {row["said"]} '
              f'sources {row["sources"]}{" (no rule: the open field)" if row["bare"] else ""} | rules {held.rules}')
    return tally


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
        optimizer = model.getOptimizer(lr=.01)
        rounds = int(os.environ.get('ROUNDS', '3'))
        for r in sorted({0, rounds}):
            if r:
                for _ in range(rounds):
                    for w in alone:
                        read(model, w, training=True, optimizer=optimizer)
                        model.End()
            print('taught', r, 'rounds of the 19 words alone')
            words = measured(model, held, alone)
            print(' row at 1.00, of', len(alone), 'words alone:', words)
            sentences = measured(model, held, ('This is Felix.',) + LESSON[1:4])
            print(' lesson sentences, of 4:', sentences)
            pairs = measured(model, held, two)
            print(' row at 1.75, of', len(two), 'two-word sentences:', pairs)
        assert words['reconstructed'] <= len(alone)
    finally:
        model.End()


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
                         torch.autograd.grad(value.sum(), [sigma.value, basic.where_entries, registry.code],
                                             allow_unused=True, retain_graph=True))
            print('in the read, gradient of', name + '.pyramid', round(float(value.sum()), 4),
                  {k: None if g is None else float(g.norm()) for k, g in zip(('sigma', 'where_entries', 'grid'), grads)})
            return wanted
        monkeypatch.setattr(SentenceField, 'charge', charged)
        read(model, 'This is Felix.', training=True, optimizer=model.getOptimizer(lr=0.))
        model.End()
        monkeypatch.setattr(SentenceField, 'charge', charge)
        field, wanted = held['field'], held['wanted']
        print('grid in an optimizer:', any(p is registry.code for g in model.getOptimizer(lr=.01).param_groups
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
