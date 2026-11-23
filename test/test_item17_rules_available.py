"""A rule of the grammar is available once it is taught (sigma-pi §19.135,
6; round 6d of item 1.7, eighteenth pass, item 1). Each rule has one
availability, by its name, held on the language space and saved with the
model; the key ``SymbolSpace.language.rulesAvailable`` names the rules
available at the start (``*`` every rule, the default). A rule not available
is on neither the compose chooser's menu nor the walk's. Measured input to
output on the lesson's model as it runs."""
import os

import torch

from test_item17_places import minimal, model_for, nominate


def _start():
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    print('start', start)


def test_the_default_is_every_rule_and_the_values_are_saved(tmp_path, monkeypatch):
    model = model_for(tmp_path, monkeypatch)
    try:
        language = model.languageSpace
        assert bool(language.rule_available.all())
        keys = [key for key in model.state_dict() if key.endswith('languageSpace.rule_available')]
        assert keys, 'the availabilities are in the model state'
        saved = model.state_dict()
        saved[keys[0]] = torch.tensor([name == 'surface' for name in language.rule_names])
        model.load_state_dict(saved)
        assert [n for n, a in zip(language.rule_names, language.rule_available.tolist()) if a] == ['surface']
        language.make_available('part')
        assert [n for n, a in zip(language.rule_names, language.rule_available.tolist()) if a] == [
            'part', 'surface']
    finally:
        model.End()


def test_with_no_rule_open_reconstruction_still_puts_back_the_words(tmp_path, monkeypatch):
    """The opening inverse precedes a grammar with no available compose rule."""
    import SentenceField
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    _start()
    model = model_for(tmp_path, monkeypatch, rules='')
    use_eager_reading(monkeypatch)
    try:
        assert not bool(model.languageSpace.rule_available.any())
        one, _, _ = minimal()
        nominate(model, one)
        language, held = model.languageSpace, dict(rules=[], errors=[], surfaces=[])
        choose, charge, reconstruct = language.choose_operation, SentenceField.charge, SentenceField.reconstruct

        def chosen(state, row_gate, **kw):
            result = choose(state, row_gate, **kw)
            c = result[0] if isinstance(result, tuple) else result
            if bool(c.applied[0]) and bool(row_gate.reshape(-1)[0]):
                held['rules'].append(int(c.kind[0]))
            return result

        def charged(model_, errors, name, carrier, field, active, **kw):
            wanted = charge(model_, errors, name, carrier, field, active, **kw)
            value = errors._value(errors._terms[name + '.pyramid'])
            held['errors'].append((name, round(float(value.detach().sum()), 4)))
            return wanted

        def reconstructed(model_, record, errors, active, **kw):
            result = reconstruct(model_, record, errors, active, **kw)
            held['surfaces'].append(bytes(kw['second'].materialize().flatten().tolist()).rstrip(b'\0'))
            return result
        monkeypatch.setattr(language, 'choose_operation', chosen)
        monkeypatch.setattr(SentenceField, 'charge', charged)
        monkeypatch.setattr(SentenceField, 'reconstruct', reconstructed)
        for word in one[:4]:
            with torch.no_grad():
                read(model, word)
                model.End()
        print('rules applied', held['rules'], '| surfaces', held['surfaces'],
              '| reconstruction errors', held['errors'])
        assert held['rules'] == []
        assert held['surfaces'] == [word.encode() for word in one[:4]]
    finally:
        model.End()


def test_one_rule_available_is_the_only_rule_on_either_menu(tmp_path, monkeypatch):
    """With one rule available, the compose chooser applies only it, and the
    walk's menu at the root admits no other rule."""
    import Generative
    from reading_fixtures import use_eager_reading
    from test_item18_expect import read
    _start()
    model = model_for(tmp_path, monkeypatch, rules='surface')
    use_eager_reading(monkeypatch)
    try:
        one, _, _ = minimal()
        nominate(model, one)
        language = model.languageSpace
        names = (*language._generate_binary_names, *language._generate_unary_names, 'STOP')
        applied, admitted = [], set()
        choose, menu = language.choose_operation, Generative.inverse_menu

        def chosen(state, row_gate, **kw):
            result = choose(state, row_gate, **kw)
            c = result[0] if isinstance(result, tuple) else result
            if bool(c.applied[0]) and bool(row_gate.reshape(-1)[0]):
                rules = language._compose_binary_rules if int(c.kind[0]) in (1, 3) else language._compose_unary_rules
                applied.append(rules[int(c.local_op[0])].method_name)
            return result

        def menus(*args, **kw):
            result = menu(*args, **kw)
            admitted.update(names[i] for i in result.admissible.any(0).nonzero().flatten().tolist())
            return result
        monkeypatch.setattr(language, 'choose_operation', chosen)
        monkeypatch.setattr(Generative, 'inverse_menu', menus)
        for word in one[:4]:
            with torch.no_grad():
                read(model, word)
                model.End()
        print('applied', applied, '| admitted on the walk\'s menus', sorted(admitted))
        assert applied and set(applied) == {'surface'}
        assert admitted <= {'surface', 'STOP'}
    finally:
        model.End()
