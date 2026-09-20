"""Learning evidence: natural wording reaches one compose/thought/generate identity."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from Language import LanguageSpace
from Layers import TernaryTruthStore, WhatInteractionMemory
from Models import BasicModel, _append_observed_meaning
from Understanding import AnswerProgram
from test_selected_relation_meaning import _program_owner


def _corpus(monkeypatch):
    cs, grammar, registry, language, _leaves, _program, _part, _whole = _program_owner(monkeypatch)
    from test_cs_symbol_table import _cs
    from Queries import GrammaticalThoughtRegistry
    cs = _cs(nS=256)
    registry = GrammaticalThoughtRegistry.install(cs, grammar)
    tokens = 'a has contains is part of equals owns person bicycle wheel car door house room boat sail box lid bag pocket tree branch cupboard shelf'.split()
    refs, rows, values = {}, {}, {}
    for token in tokens:
        identifier = cs.new_concept()
        row = cs._csw_concept_row(0, identifier)
        refs[token], rows[token] = ('sym', identifier), int(row)
        # Arbitrary dictionary atoms, no encoded meanings or identifier values.
        values[token] = registry._payload(refs[token]).detach()
        cs.remember_word_surface(row, token.encode(), object_row=row, object_id=identifier)

    def program(text):
        words = text.split()
        leaves = torch.stack([values[word] for word in words])
        return AnswerProgram(rows=torch.tensor([rows[word] for word in words]),
            word_rows=torch.tensor([rows[word] for word in words]),
            concept_ids=torch.tensor([refs[word][1] for word in words]),
            activations=torch.ones(len(words)), leaves=leaves,
            actions=torch.tensor([[0, -1, i] for i in range(len(words))]),
            targets=torch.tensor([-1]), end_state=torch.zeros(3, 8),
            lexical_forms=(None,) * len(words))

    examples = []
    for whole, part in [('car', 'door'), ('house', 'room'), ('boat', 'sail'),
                        ('box', 'lid'), ('bag', 'pocket'), ('tree', 'branch')]:
        target = registry.form('whole', refs[whole], refs[part], mode='assertive')
        canonical = program(f'a {whole} has a {part}')
        for sentence in (f'a {whole} has a {part}', f'a {whole} contains a {part}',
                         f'a {part} is part of a {whole}'):
            examples.append((program(sentence), target, canonical))
        examples.append((program(f'a {whole} equals a {whole}'),
                         registry.form('equal', refs[whole], refs[whole], mode='assertive'),
                         program(f'a {whole} equals a {whole}')))
        examples.append((program(f'a {whole} owns a {part}'), None, None))
        examples.append((program(f'a person has a {whole}'), None, None))
    return cs, registry, language, refs, rows, values, program, examples


def test_natural_parthood_is_learned_then_shared_with_thought_and_generation(monkeypatch):
    torch.manual_seed(19)
    cs, registry, language, refs, rows, values, program, examples = _corpus(monkeypatch)
    assert 'has' not in language._surface_anchors
    vocabulary = tuple(rows[word] for word in ('a', 'has', 'equals'))
    language.configure_meaning_learning(registry, vocabulary,
        torch.stack([values[word] for word in ('a', 'has', 'equals')]), hidden=48)
    held = program('a bicycle has a wheel')
    assert language.program_meaning(held, registry) is None
    optimizer = torch.optim.Adam(language.meaning_codec.parameters(), lr=.015)
    losses = []
    for step in range(240):
        optimizer.zero_grad(set_to_none=True)
        loss = language.meaning_learning_loss(examples, registry)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0] * .08, (losses[0], losses[-1])
    meaning = language.program_meaning(held, registry)
    expected = registry.form('whole', refs['bicycle'], refs['wheel'], mode='assertive')
    assert meaning is not None and meaning.role_refs == expected.role_refs
    torch.testing.assert_close(meaning.roles, expected.roles)
    from test_selected_nested_meaning import _write_selected_observation
    for text in ('a bicycle has a wheel', 'a bicycle contains a wheel',
                 'a wheel is part of a bicycle', 'a cupboard has a shelf'):
        result = language.program_meaning(program(text), registry)
        target = (registry.form('whole', refs['cupboard'], refs['shelf'], mode='assertive')
                  if 'cupboard' in text else expected)
        assert result is not None and result.role_refs == target.role_refs
        torch.testing.assert_close(result.roles, target.roles)
        for path in ('eager', 'pending', 'packed'):
            store = _write_selected_observation(language, registry, program(text), path)
            assert len(store) == 1
            assert store.row(0)['meaning'].role_refs == target.role_refs
            torch.testing.assert_close(store.row(0)['meaning'].roles, target.roles.detach())
    assert language.program_meaning(program('a bicycle owns a wheel'), registry) is None
    assert language.program_meaning(program('a person has a bicycle'), registry) is None

    cs.add_whole(refs['wheel'][1], refs['bicycle'])
    model = BasicModel(); model.spaces = []
    memory = WhatInteractionMemory(batch=1, capacity=64, detach_mode='episode')
    object.__setattr__(model, 'conceptualSpace', cs)
    object.__setattr__(model, 'languageSpace', language)
    object.__setattr__(model, 'grammatical_thoughts', registry)
    object.__setattr__(model, 'symbolSpace', SimpleNamespace(what_memory=memory,
        grammatical_thoughts=registry, ltm_store=TernaryTruthStore(8, capacity=16)))
    with model._query_boundary_scope((0,)):
        selected = model.run_selected_thought(replace(meaning, mode='interrogative'), work_budget=64)
    assert selected.result.semantic_id == 'part' and selected.support_true == 1
    emitted = language.generate_meaning(selected.meaning, registry, max_words=16)
    assert emitted is not None
    words = []
    for source, index in emitted.selections:
        if source == 'role':
            index = cs._csw_row_of(selected.meaning.role_refs[index][1])
        words.append(cs.word_surface_for_row(index).decode())
    text = ' '.join(words)
    assert text == 'a bicycle has a wheel'
    recomposed = language.program_meaning(program(text), registry)
    assert recomposed.role_refs == meaning.role_refs
    assert not emitted.truncated
    assert model._realize_thought_sentences(selected) == ('a bicycle has a wheel',)
    from Output import AnswerDerivation
    model._walk_budget = lambda: 16
    generated = model._generate_thought_words(AnswerDerivation(answer_symbol=None,
        answer_meanings=((meaning,),)))
    torch.testing.assert_close(generated[0][0], emitted.words)
    before = model.symbolSpace.ltm_store
    _append_observed_meaning(before, held.end_state, 1, meaning=meaning)
    assert before.row(0)['meaning'].role_refs == meaning.role_refs
    memory.end_what_episode()

    # Capture once: subsequent training cannot revise an earlier hard meaning.
    captured = model._program_entries(
        (torch.arange(len(held.leaves))[None], held.actions[None], held.targets[None]),
        held.leaves[None], held.rows[None], held.word_rows[None], held.activations[None],
        held.end_state[None], concept_ids=held.concept_ids[None])[0]
    assert captured.meaning_captured and captured.selected_meaning.role_refs == meaning.role_refs
    saved_state = {key: value.detach().clone() for key, value in language.state_dict().items()}
    with torch.no_grad():
        language.meaning_codec.operation_head.bias[-1] = 1000
    assert language.program_meaning(held, registry) is None
    assert language.program_meaning(captured, registry).role_refs == meaning.role_refs
    assert not captured.detached().selected_meaning.roles.requires_grad
    language.load_state_dict(saved_state)

    # Restore the learned language weights without any surface/operator table.
    restored = LanguageSpace(language._symbol_space)
    restored.restore_meaning_learning(language.state_dict(), '', registry)
    restored.load_state_dict(language.state_dict(), strict=True)
    assert restored.program_meaning(held, registry).role_refs == meaning.role_refs


def test_meaning_supervision_trains_through_runbatch_after_output(monkeypatch, tmp_path):
    from test_output_walk import _model, _capture_program_probe
    from What import What
    model = _model()
    model._tensor_peer_while_eager = True
    model._chart_compose_per_word = lambda: None
    restored = None
    try:
        with torch.no_grad():
            understood = _capture_program_probe(model, ['12 plus 1', '3 plus 4'])
        entry = understood.answer_program[0]
        model._staged_in_sub = None
        # Create the live optimizer first: lazy language parameters must join it.
        optimizer = model.getOptimizer(lr=.001)
        codec = model.configure_meaning_learning(
            (int(entry.word_rows[0]),), entry.leaves[:1], hidden=8)
        before = codec.operation_head.bias.detach().clone()
        loss_calls = []
        original = model.languageSpace.meaning_learning_loss
        def observed(examples, registry):
            assert model._last_answer_construction is not None
            assert tuple(example[0] for example in examples) == model._last_understanding.answer_program
            loss_calls.append(True)
            return original(examples, registry)
        monkeypatch.setattr(model.languageSpace, 'meaning_learning_loss', observed)
        batch = (model.inputSpace.prepInput(['12 plus 1', '3 plus 4']), torch.zeros(2, 1, 1))
        model._last_answer_construction = None
        model.runBatch(train=True, batchSize=2, split='train', optimizer=optimizer,
            batch_override=batch, questions=(What.supervised(0), What.supervised(1)),
            meaning_supervision=((None, None), (None, None)))
        assert loss_calls == [True]
        assert not torch.equal(codec.operation_head.bias, before)
        owned = [id(p) for group in optimizer.param_groups for p in group['params']]
        assert all(owned.count(id(p)) == 1 for p in codec.parameters())
        path = str(tmp_path / 'meaning.pt')
        model.save_weights(path)
        restored = _model()
        assert restored.load_weights(path, strict=True, require_match=True)
        torch.testing.assert_close(restored.languageSpace.meaning_codec.operation_head.bias,
                                   codec.operation_head.bias)
    finally:
        for item in (model, restored):
            if item is not None:
                item.End()
                item.symbolSpace.soft_reset()
