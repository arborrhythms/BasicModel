"""Conditional speech through the public answer of the lesson model with its full grammar."""

import torch

from test_item17_echo import answer_text
from What import What
from test_item17_places import model_for, nominate

PAIRS = (('say Felix', 'Felix'), ('Ciao', ''), ('Felix', ''), ('say Ciao', 'Ciao'))


def test_space_is_a_native_percept(tmp_path, monkeypatch, eager_reading):
    from SentenceLooks import Looks
    from test_item18_expect import read
    model = model_for(tmp_path, monkeypatch)
    spaces = []
    run = Looks.run

    def observed(path, arrive, **kwargs):
        field = model.perceptualSpace._forward_input
        ids, spans = field['native_indices'][0], field['native_part_spans'][0]
        store = field['percept_store']
        spaces.append([(lo, hi) for pid, (lo, hi) in zip(ids.tolist(), spans.tolist(), strict=True)
                       if pid >= 0 and store.bytes_for(pid) == b' '])
        return run(path, arrive, **kwargs)

    monkeypatch.setattr(Looks, 'run', observed)
    try:
        read(model, 'say  Felix')
        assert spaces and all(row == [(3, 4), (4, 5)] for row in spaces), spaces
    finally:
        model.End()


def command_trial(model, pairs, rounds, *, probes=None):
    """Teach paired texts and evaluate with one common, target-free inference question."""
    data = model.inputSpace.data
    data.has_supervised_outputs = True
    data.text_answers = {'train': [], 'validation': []}
    optimizer = model.getOptimizer()
    source = model.inputSpace
    question = What.inference(0, split='train')
    teacher = What.supervised(0)

    def present(text, question, train):
        with torch.set_grad_enabled(train):
            batch, _ = model.runBatch(train=train, optimizer=optimizer, batchSize=1, split='train',
                questions=(question,), batch_override=(source.prepInput([text]), torch.empty(1, 0)))
        return batch

    answers = []
    for epoch in range(rounds + 1):
        if epoch:
            for text, target in pairs:
                data.text_answers['train'] = [target]
                present(text, teacher, True)
                model.End()
        if epoch in (0, rounds):
            answers = []
            data.text_answers['train'] = []
            for text, _ in (pairs if probes is None else probes):
                batch = present(text, question, False)
                answer = answer_text(batch.outputPred)
                derivation = model._last_answer_construction.derivation
                print(epoch, repr(text), repr(answer), derivation.row_sources,
                      [len(row) for row in derivation.answer_meanings], flush=True)
                answers.append(answer)
                model.End()
    return answers


def test_the_answer_depends_on_the_spoken_request(tmp_path, monkeypatch, eager_reading):
    model = model_for(tmp_path, monkeypatch)
    try:
        nominate(model, ('say', 'Felix', 'Ciao') + tuple(text for text, _ in PAIRS))
        assert command_trial(model, PAIRS, 6) == [target for _, target in PAIRS]
    finally:
        model.End()


def test_production_operands_are_available_before_the_answer_is_scored(tmp_path, monkeypatch, eager_reading):
    from Output import emitted_meanings
    import WholePaths

    model = model_for(tmp_path, monkeypatch)
    try:
        nominate(model, ('say', 'Felix', 'say Felix'))
        owner = model._concept_owner()
        data = model.inputSpace.data
        monkeypatch.setattr(data, 'has_supervised_outputs', True)
        monkeypatch.setattr(data, 'text_answers', {'train': [], 'validation': []})
        choose, score = model._choose_selected_thought_action, WholePaths.answer_cost
        menus, scored = [], []

        def observe_choice(root, active, actions, **kwargs):
            refs = [item.request.constituents[0].role_refs[0] for item in actions
                    if item is not None and item.semantic_id == 'output' and item.request.constituents]
            menus.append({owner.interpret.lexical_form(ref[1]) for ref in refs if ref and ref[0] == 'sym'})
            return choose(root, active, actions, **kwargs)

        def observe_answer(model, construction, thoughts, questions, active, registry, **kwargs):
            emitted = {row: emitted_meanings(result) for row, result in thoughts}
            for row, actual in enumerate(construction.derivation.answer_meanings):
                expected = emitted.get(row, ())
                assert len(actual) == len(expected)
                for said, operand in zip(actual, expected, strict=True):
                    assert said.role_refs == operand.role_refs
                    torch.testing.assert_close(said.roles, operand.roles)
            scored.append(answer_text(construction.actual))
            return score(model, construction, thoughts, questions, active, registry, **kwargs)

        monkeypatch.setattr(model, '_choose_selected_thought_action', observe_choice)
        monkeypatch.setattr(WholePaths, 'answer_cost', observe_answer)
        for text, wanted in [('say Felix', {'say', 'Felix'}), ('Felix', {'Felix'})]:
            menus.clear()
            scored.clear()
            with torch.no_grad():
                result, _ = model.runBatch(train=False, optimizer=model.getOptimizer(), batchSize=1,
                    split='train', questions=(What.inference(0, split='train'),),
                    batch_override=(model.inputSpace.prepInput([text]), torch.empty(1, 0)))
            assert menus and wanted <= menus[0], menus
            assert scored == [answer_text(result.outputPred)]
            model.End()
    finally:
        model.End()
