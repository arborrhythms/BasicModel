"""Certificate 55: one time of the model, encoded in one place (rule 64)."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def _source():
    return {path.name: path.read_text() for path in (ROOT / 'bin').glob('*.py')}


def test_55a_one_clock_and_no_second_counter_is_read_as_time():
    from Memory import LTM0, STM0
    names = ('model_clock', '_ModelTimeEncoding', '_model_tick', 'when_encoding.t ', 'when_age_scale',
             'age_band', 'age_interval', 'relative_idea', 'next_time')
    for file, text in _source().items():
        for name in names:
            assert name not in text, (file, name)
    # A store names its rows by a count; its time is the model's present.
    for store in (LTM0(12, capacity=2, percept_width=8), STM0(12, 2)):
        assert 'names_issued' in dict(store.named_buffers())
        assert not any('clock' in name for name, _ in store.named_buffers())


def test_55b_every_code_of_a_time_comes_from_the_one_encoder():
    from RowTime import encoder
    from Spaces import event_when_encoding
    assert encoder() is encoder()
    assert event_when_encoding(4) is encoder()
    # No encoder of time is made anywhere else: outside the one encoder, the
    # time codecs are constructed only as zero-width carriers and in their
    # own self-tests.
    classes = ('WhenEncoding', 'DocumentWhenEncoding', 'WhenStartDurationEncoding', 'WhenRangeEncoding')
    for file, text in _source().items():
        if file == 'RowTime.py':
            continue
        tree = ast.parse(text)
        for function in ast.walk(tree):
            if not isinstance(function, ast.FunctionDef):
                continue
            for node in ast.walk(function):
                if isinstance(node, ast.Call) and getattr(node.func, 'id', None) in classes:
                    zero = ([k.value.value for k in node.keywords if k.arg == 'n_when'
                             and isinstance(k.value, ast.Constant)] == [0]
                            or [a.value for a in node.args if isinstance(a, ast.Constant)] == [0, 0])
                    assert zero or function.name == 'test', (file, function.name)


def _model(tick=0):
    from clock_fixtures import install
    from RowTime import encoder
    store = SimpleNamespace(when_start=8)
    model = install(SimpleNamespace(symbolSpace=SimpleNamespace(ltm_store=store)), tick)
    return model, encoder()


def test_55c_the_chooser_and_an_operator_receive_the_present_in_that_code(monkeypatch):
    from RowTime import compared, how_far_back
    from Spaces import ConceptualSpace
    model, clock = _model(40)
    idea = torch.cat((torch.ones(2, 8), clock.encode(torch.tensor([37, 40])), torch.ones(2, 3)), -1)
    # The chooser's reads of an idea (compose, thought, attention, the bind
    # read's keys) go through ``compared``, which takes the present there.
    read = compared(model, idea)
    torch.testing.assert_close(read[:, 8:12], how_far_back(model.present_code(), idea[:, 8:12]))
    assert clock.decode_index(read[:, 8:12]).tolist() == [3, 0]
    received = []
    code = model.present_code
    monkeypatch.setattr(model, 'present_code', lambda: received.append(1) or code())
    compared(model, idea)
    assert received
    for file, text in _source().items():
        if file in ('Language.py',):
            assert 'compared(model, state[0])' in text
        if file in ('Models.py',):
            assert 'thought_logits(replace(active, roles=compared(self, active.roles))' in text
    # An operator, the bind read, is given the present's code.
    value = torch.cat((torch.ones(1, 2, 10), clock.encode(torch.tensor([[33, 40]]))), -1)
    sub = SimpleNamespace(is_empty=lambda: False, materialize=lambda: value,
                          nWhat=6, nWhere=4, nWhen=4, whenEncoding=clock)
    content, _, stored = ConceptualSpace._bind_read(sub, 14, model.present_code())
    assert clock.decode_index(content[0, :, -4:]).tolist() == [7, 0]
    torch.testing.assert_close(stored, value)


@pytest.mark.parametrize('present', (5_000_000, 4_000_000_000))
def test_55d_the_comparator_gives_how_far_back_exactly_as_a_code(present):
    from RowTime import encoder, how_far_back
    clock = encoder()
    for back in (1, 1000, 1_000_000):
        code = how_far_back(clock.encode(present), clock.encode(present - back))
        assert int(clock.decode_index(code)) == back
        torch.testing.assert_close(code, clock.encode(back).to(code), rtol=0, atol=1e-6)


def test_55d_no_learned_reader_is_given_a_count_of_ticks():
    texts = _source()
    # The attention context gives each end of a slot's interval as a code.
    context = texts['AttentionContext.py']
    assert 'how_far_back(model.present_code(), encoder().encode(placements' in context
    assert '/ registry.capacity' not in context
    # The reference policy reads how far back a mention lies as a code, and
    # no rank of recency.
    close = texts['ReferenceClose.py']
    assert 'recency' not in close.split('def score_order')[1].split('def continuity')[0]
    assert "gap = encoder().encode(" in close
    # A question's context: no normalized position and no raw count.
    models = texts['Models.py']
    context = models.split('def _what_grammar_context')[1].split('def choose_what_slot')[0]
    assert '/ extent' not in context and 'memory_scalar' not in context


def test_55d_a_question_context_gives_its_quantities_as_codes():
    from Models import BasicModel
    from Spaces import WhereEncoding
    from What import What
    ladder = WhereEncoding(8, 4, 0)
    model = SimpleNamespace(_what_memory=lambda: SimpleNamespace(what_context=lambda question, b: dict(
        temporal=question.context_values(), input_mask=(1, 1, 1), output_mask=(1,),
        input_representations=(), output_representations=(), open_depth=2, parity=True,
        closure_pressure=0.)),
        _what_where_encoder=lambda split: ladder,
        _what_representation_summary=lambda value, device, dtype: torch.zeros(4))
    rows, _ = BasicModel._what_grammar_context(model, (What.past(5, offset=-2),),
                                               device=torch.device('cpu'), dtype=torch.float32)
    codes = rows[0, 5:33].reshape(7, 4)
    expected = ladder.encode(torch.tensor([-2., 5., 3., 3., 1., 2., 0.])).float()
    torch.testing.assert_close(codes, expected)


def test_55e_a_row_promoted_to_a_concept_keeps_its_stamp():
    from Memory import LTM0
    from MemoryConsolidation import _content
    from RowTime import encoder
    store = LTM0(12, capacity=2, percept_width=8)
    store._model_present_code = lambda: encoder().encode(9)
    row = store.append_idea(torch.ones(12))
    torch.testing.assert_close(_content(store, row)[4:8], encoder().encode(9))


def test_55f_the_code_on_the_model_follows_every_advance_of_the_clock():
    from RowTime import encoder
    model, clock = _model(0)
    for _ in range(5):
        tick = model._advance_when_time()
        assert tick == int(model.when_time)
        torch.testing.assert_close(model.present_code(), clock.encode(model.when_time).float())
    model.set_present(1_000_003)
    torch.testing.assert_close(model.present_code(), encoder().encode(1_000_003).float())
