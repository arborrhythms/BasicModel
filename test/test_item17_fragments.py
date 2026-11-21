"""Certificate 50 (sigma-pi §21, rule 58): a run of words read at one time by
the basic level, taught by System 2, and the grammar's option to take it."""
import inspect

import pytest
import torch

import FragmentRead
import IdeaRead
from test_item18_expect import lesson, read

SENTENCE = 'alpha gamma beta'


def model_with(tmp_path, monkeypatch, texts=(SENTENCE,), bias=0.):
    model = lesson(tmp_path, monkeypatch)
    from nomination_fixtures import nominated_sentences
    nominated_sentences(model, list(texts))
    model._stm_reducer().fragment_bias = bias
    return model


def kept_programs(monkeypatch):
    """Each close's kept program, its leaves' word slots and the paths' looks."""
    seen = []
    complete = IdeaRead.complete

    def capture(host, record, present, registry):
        f = inspect.currentframe().f_back
        observations, selected = f.f_locals['observations'], f.f_locals['selected']
        looks = f.f_locals['look_trials']
        program = observations[int(selected[0])]['entries'][0]
        walked = looks[int(selected[0])]
        seen.append(dict(program=program, slots=list(host._sentence_read_order[0]),
                         taken=[d['fragment'][0].nonzero().flatten().tolist() for d in walked.decisions
                                if 'fragment' in d and bool(d['fragment_taking'][0])]))
        return complete(host, record, present, registry)
    monkeypatch.setattr(IdeaRead, 'complete', capture)
    return seen


def test_50a_with_the_option_off_nothing_of_the_read_runs(tmp_path, monkeypatch):
    model = model_with(tmp_path, monkeypatch)
    try:
        assert model._stm_reducer().fragment_bias == 0.
        def never(*args, **kwargs):
            raise AssertionError('the read ran with its option off')
        for name in ('knowing', 'read', 'classes', 'constituents', 'run_at'):
            monkeypatch.setattr(FragmentRead, name, never)
        seen = kept_programs(monkeypatch)
        for _ in range(2):
            read(model, SENTENCE, training=True)
            terms = model._sentence_cost_registry._terms
            assert not any('fragment' in name for name in terms)
        assert all(not entry['taken'] for entry in seen)
        assert all(len(entry['program'].leaves) == len(entry['slots']) for entry in seen)
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_50b_no_form_row_or_symbol_is_made_for_a_run(tmp_path, monkeypatch):
    model = model_with(tmp_path, monkeypatch, bias=1.)
    try:
        from Spaces import _concept_alloc_of
        alloc = _concept_alloc_of(model._concept_owner())
        def state():
            return (alloc.next_id, len(alloc.concepts), frozenset(alloc.word_forms))
        calls = []
        for name in ('take', 'complete', 'as_leaf'):
            original = getattr(FragmentRead, name)
            def wrapped(*args, _original=original, _name=name, **kwargs):
                before = state()
                value = _original(*args, **kwargs)
                calls.append((_name, before == state()))
                return value
            monkeypatch.setattr(FragmentRead, name, wrapped)
        seen = kept_programs(monkeypatch)
        for _ in range(3):
            read(model, SENTENCE, training=True)
        assert any(entry['taken'] for entry in seen)
        assert calls and all(same for _, same in calls)
        runs = {' '.join(SENTENCE.split()[i] for i in taken) for entry in seen for taken in entry['taken']}
        print('runs taken', sorted(runs))
        assert runs and not any(form in runs or ' ' in str(form) for form in alloc.word_forms)
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_50c_the_read_has_the_composed_ideas_shape_and_teaches_the_idea_stack_alone(tmp_path, monkeypatch):
    # A probability that is never drawn: the read is taught, never taken.
    model = model_with(tmp_path, monkeypatch, bias=1e-12)
    try:
        owner = model._concept_owner()
        own = {id(p) for p in owner.idea_stack.parameters()}
        found = []
        complete = FragmentRead.complete

        def capture(host, observations, selected, present, registry, sid, looks=()):
            from Layers import Error
            program = observations[int(selected[0])]['entries'][0]
            slots = list(host._sentence_read_order[0])
            parts = FragmentRead.constituents(host.languageSpace, program, slots)
            if parts:
                at, words, seat = parts[0]
                bank = host.__dict__['_sentence_banks'][sid]
                mask = torch.zeros(1, max(words) + 1, dtype=torch.bool)
                mask[0, list(words)] = True
                field = FragmentRead.knowing(owner, bank, mask)
                idea = FragmentRead.read(owner, field)[0]
                assert idea.shape == program.operation_values[at, 2].shape
                assert not field.requires_grad
                mine = Error(row_mask=present)
                for p in host.parameters():
                    p.grad = None
                complete(host, observations, selected, present, mine, sid, looks=looks)
                assert 'expectation.fragment' in mine._terms
                mine.total().backward()
                grads = [(id(p) in own, p.grad is not None and bool(p.grad.any())) for p in host.parameters()]
                found.append(grads)
                for p in host.parameters():
                    p.grad = None
            return complete(host, observations, selected, present, registry, sid, looks=looks)
        monkeypatch.setattr(FragmentRead, 'complete', capture)
        for _ in range(3):
            read(model, SENTENCE, training=True)
        assert found
        for grads in found:
            assert any(mine and moved for mine, moved in grads)
            assert not any(moved and not mine for mine, moved in grads)
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_50d_with_the_option_on_a_run_is_one_constituent_and_the_walk_is_shorter(tmp_path, monkeypatch):
    model = model_with(tmp_path, monkeypatch)
    try:
        seen = kept_programs(monkeypatch)
        leaves = []
        as_leaf = FragmentRead.as_leaf

        def capture(host, payload, index, rows):
            value, restore = as_leaf(host, payload, index, rows)
            held = host.__dict__.get('_fragment_leaf')
            if held is not None and held['idea'] is not None and bool((rows & held['taking']).any()):
                table = getattr(host, '_occupation_word_table', None)
                leaves.append(dict(value=value[0][0, 0].detach().clone(), idea=held['idea'][0].detach().clone(),
                                   classes=held['classes'][0].clone(),
                                   table=None if table is None else table[0, int(index)].clone()))
            return value, restore
        monkeypatch.setattr(FragmentRead, 'as_leaf', capture)
        counts = {}
        for bias in (0., 1.):
            model._stm_reducer().fragment_bias = bias
            start = len(seen)
            for _ in range(3):
                read(model, SENTENCE, training=True)
            counts[bias] = [(len(e['program'].leaves), len(e['program'].actions), e['taken']) for e in seen[start:]]
        print('leaves, actions and runs taken, option off:', counts[0.])
        print('leaves, actions and runs taken, option on: ', counts[1.])
        assert all(not taken for _, _, taken in counts[0.])
        assert any(taken for _, _, taken in counts[1.])
        for n, _, taken in counts[1.]:
            # Each run taken is one leaf in place of its words.
            assert n == len(SENTENCE.split()) - sum(len(run) - 1 for run in taken)
        assert sum(a for _, a, _ in counts[1.]) < sum(a for _, a, _ in counts[0.])
        assert leaves
        for leaf in leaves:
            torch.testing.assert_close(leaf['value'], leaf['idea'].to(leaf['value']))
            if leaf['table'] is not None:
                torch.testing.assert_close(leaf['table'], leaf['classes'].to(leaf['table']))
        print('the word table was', 'present' if leaves[0]['table'] is not None else 'absent',
              'at the leaves taken')
    finally:
        model.End(); model.symbolSpace.soft_reset()


def test_50e_learning_how_far_the_read_lies_from_the_composed_idea(tmp_path, monkeypatch):
    """Recorded, not repaired: the read idea's distance from the composed
    idea of its constituent, and from another sentence's, before and after
    teaching."""
    texts = ('alpha gamma', 'beta delta')
    model = model_with(tmp_path, monkeypatch, texts=texts, bias=1e-12)
    try:
        owner = model._concept_owner()
        record = {text: [] for text in texts}
        complete = FragmentRead.complete
        current = {}

        def capture(host, observations, selected, present, registry, sid, looks=()):
            program = observations[int(selected[0])]['entries'][0]
            parts = FragmentRead.constituents(host.languageSpace, program, list(host._sentence_read_order[0]))
            if parts:
                at, words, _ = parts[-1]
                bank = host.__dict__['_sentence_banks'][sid]
                mask = torch.zeros(1, max(words) + 1, dtype=torch.bool)
                mask[0, list(words)] = True
                with torch.no_grad():
                    idea = FragmentRead.read(owner, FragmentRead.knowing(owner, bank, mask))[0]
                record[current['text']].append((idea, program.operation_values[at, 2].detach().clone()))
            return complete(host, observations, selected, present, registry, sid, looks=looks)
        monkeypatch.setattr(FragmentRead, 'complete', capture)
        for _ in range(8):
            for text in texts:
                current['text'] = text
                read(model, text, training=True)
        if not all(record.values()):
            pytest.fail(f'a sentence had no constituent of two words: {[len(v) for v in record.values()]}')
        def distance(a, b):
            return float((a - b).square().mean())
        for text, other in (texts, texts[::-1]):
            first, last = record[text][0], record[text][-1]
            print(f'{text!r}: read against its composed idea, first {distance(*first):.4f}, '
                  f'last {distance(*last):.4f}; against {other!r}\'s composed idea, '
                  f'last {distance(last[0], record[other][-1][1]):.4f}')
    finally:
        model.End(); model.symbolSpace.soft_reset()
