"""Item 1.7, round 4: symbols and the orders above.

Certificates 51 and 54 of the sigma-pi specification, §21, at order zero:
Pi, Not, Sigma. The Not makes each row's two values one signed value and
holds the sign of every entry; every weight of the Sigma is at zero or
above (Alec, 2026-11-19, in place of rule 59's two channels); a Not entry is
made by the gradient or by the Not's teaching; and the Sigma reads the basic
percepts' rows and never a percept's code (rule 62). The certificates
of the round's other steps join this file.
"""
import json
from pathlib import Path

import pytest
import torch

from Layers import NotLayer3, SparseSigmaLayer3
from OrderZero import read_word


def _read_lesson_at_start(sentences):
    """The lesson's model after ``sentences`` sentences read with a learning
    rate of zero: every entry of the order-zero Sigma is as its writers
    made it, at its start. Returns the model, the patch to undo and every
    percept field the order-zero Sigma read."""
    from pytest import MonkeyPatch
    from reading_fixtures import use_eager_reading
    from Spaces import ConceptualSpace
    patch = MonkeyPatch()
    use_eager_reading(patch)
    fields = []
    original = ConceptualSpace.cs_read_percept_field

    def kept(self, perception, **kwargs):
        fields.append(perception)
        return original(self, perception, **kwargs)
    patch.setattr(ConceptualSpace, 'cs_read_percept_field', kept)
    import Language
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    model = BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]
    model._tensor_peer_while_eager = True
    optimizer = model.getOptimizer(lr=0.)
    curriculum = json.loads(Path('data/corpus-lessons.json').read_text())
    texts, targets = stage_documents(TheData, curriculum['stages'][0])
    admit_training_forms(model, texts, targets)
    count = 0

    class Complete(Exception):
        pass

    def batch(*args, **kwargs):
        nonlocal count
        count += 1
        if count == sentences:
            raise Complete
    try:
        present(model, split='train', optimizer=optimizer, batch_size=1, after_batch=batch)
    except Complete:
        pass
    return model, patch, fields


@pytest.fixture(scope='module')
def lesson_at_start():
    model, patch, fields = _read_lesson_at_start(16)
    yield model._concept_owner(), fields
    model.End()
    patch.undo()


# Certificate 51, in place of the two channels: Pi, Not, Sigma.

def test_51a_no_weight_of_the_order_zero_sigma_is_below_zero(lesson_at_start):
    """(a) No weight of the order-zero Sigma is below zero: in the lesson's
    model at its start, and after training on the three learning cases of
    certificate 23. The Not entries each case ends with are printed."""
    from test_item17_order_zero import _taught, _with_rows, _not_entries
    owner, fields = lesson_at_start
    sigma = owner.order_zero
    n = sigma.nnz
    assert n and bool((sigma.weight[:n] >= 0).all())
    rows = _with_rows(((0, 1, 2), (3, 4, 5), (6, 7, 8)), ((0, 1), (0, 2)))
    words = _with_rows(((0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)), ((0, 2), (0, 3), (1, 2), (1, 3)))
    report = {}
    for name, cases, rs, project in (
            ('patterning', {(0,): 1., (1,): 1., (0, 1, 3): 0.}, rows, False),
            ('inhibition', {(0,): 1., (0, 2, 4): 0., (1,): 1.}, rows, False),
            ('xor', {(0, 2, 4): 0., (0, 3, 5): 1., (1, 2, 6): 1., (1, 3, 7): 0.}, words, True)):
        order, _, _ = _taught(cases, rows=rs, project=project, seed=0)
        weight = order.sigma.weight.detach()[:order.sigma.nnz]
        assert bool((weight >= 0).all())
        report[name] = dict(least_weight=float(weight.min()), not_entries=_not_entries(order, .01))
    print(dict(lesson_entries=n, lesson_least=float(sigma.weight[:n].min()), taught=report))


def test_51b_with_no_not_entry_raising_a_row_never_lowers_a_concept():
    """(b) With no Not entry, raising any row's signed value never lowers
    any concept: drawn weights at zero or above, drawn readings, each row
    raised in turn."""
    torch.manual_seed(int.from_bytes(__import__('os').urandom(4), 'little'))
    rows, concepts = 6, 4
    not_, sigma = NotLayer3(rows), SparseSigmaLayer3(rows, concepts, 32)
    for c in range(concepts):
        for r in torch.randperm(rows)[:3].tolist():
            sigma.add(c, r, float(torch.rand(())))
    assert not bool(NotLayer3.entries(sigma).any())
    lanes = torch.rand(rows, 20, 2)
    y = sigma(not_(lanes), range(rows), range(concepts))
    for r in range(rows):
        raised = lanes.clone()
        raised[r, :, 0] = (raised[r, :, 0] + .3).clamp(max=1.)
        raised[r, :, 1] = (raised[r, :, 1] - .3).clamp(min=0.)
        assert bool((sigma(not_(raised), range(rows), range(concepts)) >= y - 1e-7).all())


def test_51c_the_not_gives_for_less_against():
    """(c) A row read for gives 1, read against -1, read both 0; read
    through a Not entry, the opposite."""
    signed = NotLayer3(1)(torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]]))
    print(dict(read_for=float(signed[0, 0]), read_against=float(signed[0, 1]), read_both=float(signed[0, 2])))
    assert signed[0].tolist() == [1., -1., 0.]
    assert NotLayer3.exchange(signed, torch.tensor(-1.))[0].tolist() == [-1., 1., 0.]


def test_51d_a_not_entry_is_made_by_the_gradient_and_by_omission():
    """(d) A concept started on its row (a for entry at one), taught to be
    present where the row is read against and absent where it is read for,
    comes by the gradient alone to read the row through a Not entry: its
    entry is carried through zero, and its weight grows from there. With one
    signed value for the row it cannot read zero on one side and one on the
    other: it settles between, the two readings each other's opposite. And
    an omission makes Not entries, with no step of the gradient: in "A
    alone, not A with X", on X and AX, so X alone reads below zero."""
    not_, sigma = NotLayer3(1), SparseSigmaLayer3(1, 1, 1)
    sigma.add(0, 0)
    lanes = torch.tensor([[[1., 0.], [0., 1.]]])  # read for, read against
    target = torch.tensor([0., 1.])
    read = lambda: sigma(not_(lanes), [0], [0])[0]
    entry = lambda: dict(entry=sigma.entries(0), weight=float(sigma.weight[0].detach()), reads=read().detach().tolist())
    before = entry()
    assert before['entry'] == [(0, 1., False)]
    optimizer = torch.optim.Adam(sigma.parameters(), lr=.05)
    for _ in range(500):
        optimizer.zero_grad()
        ((read() - target) ** 2).mean().backward()
        optimizer.step()
    after = entry()
    print(dict(before=before, after=after))
    # One signed value for the row: the concept reads the row read for as
    # the opposite of the row read against, so it settles between the two
    # targets, present where the row is read against and denied where it
    # is read for.
    assert after['entry'][0][2] and after['weight'] > 0
    assert after['reads'][1] > 0 > after['reads'][0]
    # The omission: a concept with for entries on A, X and AX (rows 0, 1, 2)
    # is read where A, X and AX are present and does not come; it comes
    # where A alone is present. The Not makes X and AX Not entries, with no
    # step of the gradient.
    not_, sigma = NotLayer3(3), SparseSigmaLayer3(3, 1, 3)
    for row in range(3):
        sigma.add(0, row)
    lanes = torch.tensor([[[1., 0.], [1., 0.]], [[0., 0.], [1., 0.]], [[0., 0.], [1., 0.]]])  # [row, reading, 2]
    expected = sigma(not_(lanes), [0, 1, 2], [0])[0].detach()
    made = not_.teach_omission(sigma, 0, lanes, [0, 1, 2], came=torch.tensor([True, False]), expected=expected)
    print(dict(expected=expected.tolist(), made_by_omission=made, entries=sigma.entries(0)))
    assert made == [1, 2] and [n for _, _, n in sigma.entries(0)] == [False, True, True]
    assert float(sigma(not_(torch.tensor([[[1., 0.]]])), [1], [0])[0, 0]) < 0  # X alone


def test_a_checkpoint_of_the_dense_sigma_is_refused_with_the_reason():
    """A Sigma of order zero saved as a dense matrix over the percept code
    has no reading on rows: it is refused, with the reason."""
    sigma = SparseSigmaLayer3(4, 2, 8)
    saved = {'weight': torch.rand(6, 2), 'against': torch.rand(6, 2)}
    with pytest.raises(ValueError, match='no reading on the rows of the first Pi'):
        sigma.load_state_dict(saved, strict=False)


# Certificate 54: the Sigma of order zero reads rows (rule 62).

ORDER_ZERO_READERS = {
    'Layers.py': ('NotLayer3', 'SparseSigmaLayer3'),
    'OrderZero.py': ('read_word',),
    'NumericPercepts.py': ('read',),
    'Spaces.py': ('order_zero_reading', 'install_order_zero', 'order_zero_field', '_start_order_zero_entry',
                  'cs_read_percept_field', 'add_concept_feature'),
    'Omission.py': ('concept_teacher',),
}
#: What reads or makes a percept's code (attributes of the basic percepts
#: and their code), or joins codes for a Sigma (functions).
CODE_ATTRIBUTES = {'codes', 'against_codes', 'code', 'read_codes', 'read_pair', 'lanes', 'column',
                   'member_column', 'value_code', 'collection_code', 'part_codes', 'whole_codes',
                   'content_width', 'order_zero_input', 'follow_members', 'combine'}
CODE_FUNCTIONS = {'combine', 'member_column', 'follow_members', 'order_zero_input'}


def test_54a_no_function_that_computes_a_concept_of_order_zero_takes_a_percepts_code():
    """(a) Every function (or class) through which a concept of order zero is
    computed or written names nothing that reads or makes a percept's code,
    or joins codes; and the dense Sigma, its joined vector and its columns
    are named nowhere in the source."""
    import ast
    root = Path(__file__).resolve().parents[1] / 'bin'
    found = {}
    for file, names in ORDER_ZERO_READERS.items():
        text = (root / file).read_text()
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                used = ({n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)} & CODE_ATTRIBUTES
                        | {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & CODE_FUNCTIONS)
                found[(file, node.name)] = sorted(used)
    print(found)
    assert {(f, n) for f, names in ORDER_ZERO_READERS.items() for n in names} <= set(found)
    assert not any(found.values()), found
    gone = ('OrderZeroSigma', 'member_column', 'follow_members', 'order_zero_input', 'read_pair',
            'read_codes', '_follow_moved')
    import re
    named = {path.name: [g for g in gone if re.search(rf'\b{g}\b', path.read_text())]
             for path in root.glob('*.py')}
    assert not any(named.values()), {k: v for k, v in named.items() if v}


def test_54c_where_a_forms_holder_stands_in_its_place_the_concept_reads_zero():
    """(c) The concept of ``archer`` reads its own row at tanh(1) and reads
    zero on ``archery``, whose row of ``archer`` reads for 1 and against 1,
    which the Not makes one signed value, 0. ``is`` on ``this`` likewise.
    Read through the first Pi, the Not and the Sigma of order zero's
    classes, one concept on each row, at its start."""
    from test_item17_code import coded, _read_form
    words = (b'archer', b'archery', b'is', b'this', b'reach')
    identity, basic, book, rows = coded(words)
    not_ = NotLayer3(basic.capacity)
    sigma = SparseSigmaLayer3(basic.capacity, len(words), 2 * len(words))
    concept = {w: k for k, w in enumerate(words)}
    for w in words:
        sigma.add(concept[w], rows[w])
    inputs = [rows[w] for w in words]
    def reads(c, form):
        lanes = _read_form(basic, book, form, inputs)
        return float(sigma(not_(lanes[:, None]), inputs, [concept[c]])[0, 0].detach())
    table = {f'{c.decode()} on {f.decode()}': reads(c, f)
             for c, f in ((b'archer', b'archer'), (b'archer', b'archery'), (b'archer', b'reach'),
                          (b'is', b'is'), (b'is', b'this'))}
    print(table)
    one = float(torch.tanh(torch.tensor(1.)))
    assert table['archer on archer'] == pytest.approx(one, abs=1e-6)
    assert table['is on is'] == pytest.approx(one, abs=1e-6)
    assert table['archer on archery'] == 0. and table['is on this'] == 0.
    assert table['archer on reach'] == 0.


def test_54bd_a_words_concept_reads_its_row_and_the_teacher_tells_the_sentences_apart(tmp_path, monkeypatch):
    """(b) In the model of the expectation's test, with ``alpha gamma`` and
    ``beta delta`` read until their words are nominated, each word's concept
    reads one before the squash where its row is present and zero on every
    reading of the other sentence, whose letters it shares (the concept of
    ``alpha`` reads zero on "beta delta"). (d) With the second word omitted,
    the expectation's teacher (``Omission.concept_teacher``) gives the words
    of the sentence read and zero to the words of the other."""
    import Omission
    from Spaces import ConceptualSpace
    from test_item18_expect import lesson, read, omit_second
    from nomination_fixtures import readings_before_nomination
    from PrependedSymbols import word_view
    model = lesson(tmp_path, monkeypatch)
    try:
        examples = ('alpha gamma', 'beta delta')
        concepts = []
        for text in examples:
            readings_before_nomination(model, lambda: read(model, text))
            read(model, text)
            concepts.extend(int(c) for c in word_view(model.inputSpace, 'concept_rows')[0, :2])
        owner = model._concept_owner()
        assert all(c >= 0 for c in concepts)
        seen = {}
        original = ConceptualSpace.cs_read_percept_field

        def kept(self, perception, **kwargs):
            if len(perception.keys):
                inputs = [key[1] for key in perception.keys]
                field = self.order_zero_field(perception.values.detach(), inputs, concepts).detach()
                seen.setdefault(model._reading_text, []).append(field[..., 0] - field[..., 1])
            return original(self, perception, **kwargs)
        monkeypatch.setattr(ConceptualSpace, 'cs_read_percept_field', kept)
        for text in examples:
            model._reading_text = text
            read(model, text)
        one = float(torch.tanh(torch.tensor(1.)))
        table = {}
        for k, text in enumerate(examples):
            y = torch.cat([v.flatten(1) for v in seen[text]], 1)  # [concept, readings]
            mine, other = (slice(0, 2), slice(2, 4)) if k == 0 else (slice(2, 4), slice(0, 2))
            table[text] = dict(own=y[mine].amax(1).tolist(), other_greatest=float(y[other].abs().max()))
            assert all(v == pytest.approx(one, abs=1e-5) for v in y[mine].amax(1).tolist())
            assert float(y[other].abs().max()) == 0.
        monkeypatch.setattr(ConceptualSpace, 'cs_read_percept_field', original)
        taught = []
        teacher = Omission.concept_teacher

        def kept_teacher(model, **kwargs):
            reading, hidden = teacher(model, **kwargs)
            if reading is not None:
                taught.append(reading[0, concepts, 0].tolist())
            return reading, hidden
        monkeypatch.setattr(Omission, 'concept_teacher', kept_teacher)
        omit_second(monkeypatch, model)
        optimizer = model.getOptimizer(lr=.01)
        for text in examples:
            read(model, text, training=True, optimizer=optimizer)
        print(dict(concepts=concepts, readings=table, teacher=taught))
        assert len(taught) == 2
        for k, values in enumerate(taught):
            mine, other = (values[:2], values[2:]) if k == 0 else (values[2:], values[:2])
            assert min(mine) > .5 and max(other) == 0.
    finally:
        model.End(); model.symbolSpace.soft_reset()
