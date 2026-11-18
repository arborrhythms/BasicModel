"""The provisional shared match and its menu operand (§70.11)."""
from types import SimpleNamespace

import pytest
import torch

from ConceptualMatch import match
from IdentityRows import meet_ratios


def bank(form_width):
    return SimpleNamespace(mereology=None if form_width is None else
                           SimpleNamespace(percept_event_width=form_width))


@pytest.mark.parametrize('form_width', [None, 0, 4, 12])
@pytest.mark.parametrize('candidate_width', [8, 12])
def test_reference_values_and_operand_gradients_equal_the_previous_read(form_width, candidate_width):
    query = torch.rand(2, 1, 12, requires_grad=True)
    rows = torch.rand(1, 3, candidate_width, requires_grad=True)
    old_query, old_rows = query.clone(), rows.clone()
    if form_width is not None:
        old_query[..., :form_width] = 0
        old_rows[..., :form_width] = 0
    expected = meet_ratios(old_query, old_rows)
    actual = match(query, rows, codebook=bank(form_width))
    for after, before in zip(actual, expected):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    old_grad = torch.autograd.grad(sum(x.sum() for x in expected), (query, rows), retain_graph=True)
    new_grad = torch.autograd.grad(sum(x.sum() for x in actual), (query, rows))
    for after, before in zip(new_grad, old_grad):
        torch.testing.assert_close(after, before, rtol=0, atol=0)


@pytest.mark.parametrize('form_width', [None, 4, 12])
@pytest.mark.parametrize('lane', [0, 1])
def test_field_values_and_operand_gradients_equal_the_previous_read(form_width, lane):
    query = torch.rand(1, 2, 3, 12, requires_grad=True) * 2 - 1
    rows = torch.rand(4, 1, 1, 12, requires_grad=True)
    rows = rows * torch.tensor([1., 1., 0., 0.])[:, None, None, None]
    width = 12 if form_width is None else form_width
    expected = meet_ratios((query[..., :width] if lane == 0 else -query[..., :width]).clamp_min(0),
                           rows[..., :width].clamp_min(0))
    if width < 12:
        half = (12-width)//2
        meaning = rows[..., width:width+half].clamp_min(0)
        old = meet_ratios(query[..., width+lane*half:width+(lane+1)*half].clamp_min(0), meaning)
        expected = tuple(torch.where(meaning.sum(-1) > 0, a, b) for a, b in zip(old, expected))
    actual = match(query, rows, codebook=bank(form_width), lane=lane)
    for after, before in zip(actual, expected):
        torch.testing.assert_close(after, before, rtol=0, atol=0)
    old_grad = torch.autograd.grad(sum(x.sum() for x in expected), (query, rows), retain_graph=True)
    new_grad = torch.autograd.grad(sum(x.sum() for x in actual), (query, rows))
    for after, before in zip(new_grad, old_grad):
        torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_empty_content_has_no_vacuous_coverage_and_poles_are_not_netted():
    rows = torch.tensor([[0., 0.], [.5, .2], [.2, .5]])
    for query in (torch.zeros(2), torch.tensor([.5, .2])):
        actual = match(query, rows)
        for after, before in zip(actual, meet_ratios(query, rows)):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
    a, b = match(torch.tensor([.5, .2]), rows)
    assert a[0] == b[0] == 0 and a[1] == b[1] == 1
    assert a[2] < 1 and b[2] < 1


def word_case(paired):
    from Memory import STMChunks as C
    from Interpret import InterpretLayer
    width = 12 if paired else 10
    codebook = bank(10 if paired else None)
    owner = SimpleNamespace(similarity_codebook=codebook)
    interpreter = InterpretLayer(conceptualSpace=owner)
    atoms = torch.zeros(1, 1, width)
    atoms[..., :2] = .2
    if paired:
        atoms[..., 10:] = .2
    candidate = torch.zeros(width)
    candidate[:2] = .8
    if paired:
        candidate[10:] = .8
    state = C.empty(1, 8, width)
    state, seat = C.deposit(state, candidate[None], torch.tensor([True]), identity=7, order=1)
    word = candidate.clone()
    word[:2] = torch.tensor([.9, .1])
    state, physical = C.deposit(state, word[None], torch.tensor([True]),
        identity=11, order=0, stack=True, activation=torch.ones(1))
    state = list(state)
    state[C.TRUTH][0, physical] = 1
    state = tuple(state)
    inputs = SimpleNamespace(_ar_word_object_ids=torch.tensor([[5]]),
        _ar_word_object_rows=torch.tensor([[1]]), _ar_word_object_orders=torch.tensor([[1]]),
        _ar_word_object_atoms=atoms)
    return state, physical, seat, word, candidate, inputs, interpreter


def word_menu(state, word, inputs, interpreter, *, word_source=-1):
    from ReferenceContext import prepare_operands
    integers = torch.zeros(1, 2, dtype=torch.long)
    return prepare_operands(word[None, None].expand(1, 2, -1), integers+11, integers, integers,
        orders=integers, rules=(SimpleNamespace(method_name='bind'),), unary_rules=(),
        bank=SimpleNamespace(interpreter=interpreter), live=None,
        active=torch.tensor([[True, False]]), chunk_state=state, source=integers,
        word_sources=integers+word_source, word_inputs=inputs, word_interpreter=interpreter)


@pytest.mark.parametrize('paired', [False, True])
def test_menu_values_the_staged_object_and_keeps_the_word_operand(paired):
    from ReferenceContext import bind_content
    from Memory import STMChunks as C
    state, physical, seat, word, candidate, inputs, interpreter = word_case(paired)
    result = bind_content(word[None, None], torch.tensor([[0]]), word_source=torch.tensor([[-1]]),
        inputs=inputs, interpreter=interpreter, chunks=state, physical=physical[:, None])
    expected = interpreter.activate(inputs._ar_word_object_atoms, torch.ones(1, 1), evidence=torch.ones(1, 1, 2))
    torch.testing.assert_close(result.value, expected, rtol=0, atol=0)
    assert result.order.item() == 1 and result.object_id.item() == 5
    assert result.found.all() and not result.held.any()
    assert not result.value[..., 2:10].any()
    menu = word_menu(state, word, inputs, interpreter)
    assert menu['binary_valid'][0, 0, seat+1]
    torch.testing.assert_close(menu['left'][0, 0, int(seat)+1], word, rtol=0, atol=0)
    assert menu['binary_requested_orders'][0, 0, int(seat)+1].tolist() == [0, 1]
    # A phrase uses the read's vector even when the same staged word exists.
    phrase = bind_content(word[None, None], torch.tensor([[0]]), word_source=torch.tensor([[2]]),
        inputs=inputs, interpreter=interpreter, chunks=state, physical=physical[:, None])
    torch.testing.assert_close(phrase.value[0, 0], word, rtol=0, atol=0)
    assert phrase.order.item() == 0 and not phrase.single.any()


def test_object_lookup_uses_the_same_timeline_and_reader_as_refresh():
    from ReferenceContext import bind_content
    from Memory import STMChunks as C
    state, physical, seat, word, candidate, inputs, interpreter = word_case(True)
    state, held = C.deposit(state, (candidate*.5)[None], torch.tensor([True]), identity=5, order=1)
    before = tuple(value.clone() for value in state)
    result = bind_content(word[None, None], torch.tensor([[0]]), word_source=torch.tensor([[-1]]),
        inputs=inputs, interpreter=interpreter, chunks=state, physical=physical[:, None])
    assert result.held.all()
    torch.testing.assert_close(result.value[0, 0], state[C.BUFFER][0, int(held)], rtol=0, atol=0)
    identified = list(state)
    identified[C.IDENTITY] = C.write(state[C.IDENTITY], physical, torch.tensor([5]), torch.tensor([True]))
    _, refreshed = C.refresh(tuple(identified), physical[:, None])
    torch.testing.assert_close(result.value, refreshed, rtol=0, atol=0)
    for actual, old in zip(state, before):
        torch.testing.assert_close(actual, old, rtol=0, atol=0)
    altered = list(state)
    altered[C.TIMELINE] = state[C.TIMELINE].clone()
    altered[C.TIMELINE][0, held] = 1
    result = bind_content(word[None, None], torch.tensor([[0]]), word_source=torch.tensor([[-1]]),
        inputs=inputs, interpreter=interpreter, chunks=tuple(altered), physical=physical[:, None])
    assert not result.held.any()


def test_unstaged_word_has_no_content_offer():
    state, _, _, word, _, inputs, interpreter = word_case(False)
    inputs._ar_word_object_rows.fill_(-1)
    menu = word_menu(state, word, inputs, interpreter)
    assert menu['binary_valid'][0, 0, 0]
    assert not menu['binary_valid'][0, 0, 1:].any()


def test_object_menu_preserves_code_presence_and_pair_gradients():
    from Memory import STMChunks as C
    from ReferenceContext import bind_content
    state, physical, _, word, _, inputs, interpreter = word_case(True)
    inputs._ar_word_object_atoms.requires_grad_()
    state = list(state)
    state[C.ACTIVATIONS] = state[C.ACTIVATIONS].clone().requires_grad_()
    state[C.TRUTH] = state[C.TRUTH].clone().requires_grad_()
    state = tuple(state)
    actual = bind_content(word[None, None], torch.tensor([[0]]), word_source=torch.tensor([[-1]]),
        inputs=inputs, interpreter=interpreter, chunks=state, physical=physical[:, None]).value
    expected = interpreter.activate(inputs._ar_word_object_atoms,
        C.gather(state[C.ACTIVATIONS], physical[:, None]),
        evidence=C.gather(state[C.TRUTH], physical[:, None]))
    operands = (inputs._ar_word_object_atoms, state[C.ACTIVATIONS], state[C.TRUTH])
    before = torch.autograd.grad(expected.sum(), operands, retain_graph=True)
    after = torch.autograd.grad(actual.sum(), operands)
    for a, b in zip(after, before):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.any()


def test_off_stack_symbol_reader_has_refresh_precedence():
    from Memory import STMChunks as C
    from ReferenceContext import bind_content
    state, physical, _, word, candidate, inputs, interpreter = word_case(True)
    state, earlier = C.deposit(state, (candidate*.5)[None], torch.tensor([True]), identity=5, order=1)
    state, preferred = C.deposit(state, (candidate*.75)[None], torch.tensor([True]), identity=5, order=1)
    state = list(state)
    state[C.REFS] = state[C.REFS].clone()
    state[C.REFS][0, int(earlier), 0, 0] = C.OCCURRENCE
    state[C.REFS][0, int(preferred), 0, 0] = C.SYMBOL
    state = tuple(state)
    result = bind_content(word[None, None], torch.tensor([[0]]), word_source=torch.tensor([[-1]]),
        inputs=inputs, interpreter=interpreter, chunks=state, physical=physical[:, None])
    torch.testing.assert_close(result.value[0, 0], state[C.BUFFER][0, int(preferred)], rtol=0, atol=0)
    identified = list(state)
    identified[C.IDENTITY] = C.write(state[C.IDENTITY], physical, torch.tensor([5]), torch.tensor([True]))
    _, refreshed = C.refresh(tuple(identified), physical[:, None])
    torch.testing.assert_close(result.value, refreshed, rtol=0, atol=0)


def test_reference_match_readers_use_the_shared_function():
    import ast
    import inspect
    import textwrap
    from Memory import STMChunks, MediumTermMemory
    from ReferenceClose import ReferenceClose
    import MemoryLink
    import FieldProjection
    readers = (STMChunks.bind_menu, MediumTermMemory.merge, ReferenceClose.score_order,
               ReferenceClose.individuals, MemoryLink.menu, MemoryLink.link, FieldProjection.stimulate)
    for reader in readers:
        tree = ast.parse(textwrap.dedent(inspect.getsource(reader)))
        names = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
        assert 'meet_ratios' not in names
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'match']
        assert calls, reader.__qualname__
        for call in calls:
            assert not any(isinstance(node, ast.Call) and
                           (isinstance(node.func, ast.Name) and node.func.id == 'coordinates'
                            or isinstance(node.func, ast.Attribute) and node.func.attr == 'coordinates')
                           for arg in call.args for node in ast.walk(arg))
    assert 'meaning_start' not in inspect.signature(STMChunks.bind_menu).parameters
    assert 'percept_band' not in inspect.signature(STMChunks.bind_menu).parameters
