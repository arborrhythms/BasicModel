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


def test_51c_the_not_offers_the_value_and_the_not_value():
    """(c) The Not offers each row two options: its value, for less
    against (read for 1, read against -1, read both 0), and its not-value,
    the pair exchanged (-1, 1, 0). An entry at or above zero reads the
    value, one below zero the not-value."""
    options = NotLayer3(1)(torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]]))
    print(dict(value=options[0, :, 0].tolist(), not_value=options[0, :, 1].tolist()))
    assert options[0, :, 0].tolist() == [1., -1., 0.]
    assert options[0, :, 1].tolist() == [-1., 1., 0.]
    assert NotLayer3.choose(options, torch.tensor(1.))[0].tolist() == [1., -1., 0.]
    assert NotLayer3.choose(options, torch.tensor(-1.))[0].tolist() == [-1., 1., 0.]


def test_51c_a_concept_of_presence_and_one_of_absence_over_one_row():
    """Two concepts over one row: one reads its value, one its not-value,
    and both stand; neither row nor concept is limited to one of the other."""
    not_, sigma = NotLayer3(2), SparseSigmaLayer3(2, 3, 8)
    sigma.add(0, 0)
    not_.make(sigma, 1, 0)
    sigma.add(2, 0), sigma.add(2, 1)
    lanes = torch.tensor([[[1., 0.]], [[1., 0.]]])
    y = sigma(not_(lanes), [0, 1], [0, 1, 2])[:, 0]
    t = float(torch.tanh(torch.tensor(1.)))
    assert y[0].item() == pytest.approx(t) and y[1].item() == pytest.approx(-t)
    assert y[2].item() == pytest.approx(float(torch.tanh(torch.tensor(2.))))
    assert sigma.rows_reading(0) == [0, 1, 2]


def test_51d_a_not_entry_is_made_by_the_gradient():
    """(d) A concept started on its row (an entry on its value, at one),
    taught to be present where the row is read against and absent where it
    is read for, comes by the gradient alone to read the row's not-value:
    its entry is carried through zero, and its weight grows from there. The
    value and the not-value of one row are each other's opposite, so it
    cannot read zero on one side and one on the other: it settles between,
    the two readings each other's opposite. A Not entry is otherwise made
    only by a declared Not member (``NotLayer3.make``)."""
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
    assert after['entry'][0][2] and after['weight'] > 0
    assert after['reads'][1] > 0 > after['reads'][0]
    assert not hasattr(NotLayer3, 'teach_omission')


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


# The sparse Sigma's read costs the entries it reads (rule 19): the same
# values and gradients as the read it replaced, which passed over every
# written entry with an index as long as the inputs' capacity.

class _PresentEntryRead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, signed):
        ctx.save_for_backward(value, signed)
        return torch.where(value < 0, -signed, signed) * value.abs()

    @staticmethod
    def backward(ctx, gradient):
        value, signed = ctx.saved_tensors
        return gradient * signed, gradient * value


def _present_forward(sigma, signed, inputs, rows):
    """The read of 1f58fe1511, over every written entry."""
    inputs = torch.as_tensor(inputs, dtype=torch.long).reshape(-1)
    rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
    shape = (len(rows), *signed.shape[1:])
    n = sigma.nnz
    if not n or not len(rows) or not len(inputs):
        return signed.new_zeros(shape)
    position = torch.full((sigma.nInput,), -1, dtype=torch.long)
    known = (inputs >= 0) & (inputs < sigma.nInput)
    position[inputs.clamp(0, sigma.nInput - 1)[known]] = torch.arange(len(inputs))[known]
    valid = (rows >= 0) & (rows < sigma.nOutput)
    wanted = torch.unique(rows[valid])
    slot = torch.full((sigma.nOutput,), -1, dtype=torch.long)
    slot[wanted] = torch.arange(len(wanted))
    at = slot[sigma.entry_row[:n]]
    read = position[sigma.entry_input[:n]]
    keep = ((at >= 0) & (read >= 0)).to(signed.dtype)
    value = (sigma.value[:n].to(signed) * keep).reshape(n, *([1] * (signed.ndim - 1)))
    terms = _PresentEntryRead.apply(value.expand(n, *signed.shape[1:]), signed.index_select(0, read.clamp_min(0)))
    total = signed.new_zeros(max(len(wanted), 1), *shape[1:]).index_add(0, at.clamp_min(0), terms)
    y = torch.tanh(total)
    index = slot[rows.clamp(0, sigma.nOutput - 1)]
    held = valid & (index >= 0)
    y = y.index_select(0, index.clamp_min(0))
    return y * held.reshape(-1, *([1] * (y.ndim - 1))).to(y)


@pytest.mark.parametrize('nots', (0, .4))
def test_the_sparse_sigma_reads_and_learns_as_the_read_it_replaced(nots):
    """Random entries (with a share ``nots`` of Not entries, made by the
    Not), random rows read and asked: the read by the compressed rows gives
    the values and the gradients, in the entries and in the rows' values,
    of the read over every entry; entries at zero included. Added in many
    steps, so the compressed part and the entries added since are both
    read."""
    torch.manual_seed(int.from_bytes(__import__('os').urandom(4), 'little'))
    P, C = 40, 30
    not_, sigma = NotLayer3(P), SparseSigmaLayer3(P, C, 600)
    for _ in range(500):
        c, r = int(torch.randint(C, ())), int(torch.randint(P, ()))
        if float(torch.rand(())) < nots:
            not_.make(sigma, c, r, float(torch.rand(())))
        else:
            sigma.add(c, r, float(torch.rand(())) * (float(torch.rand(())) > .1))
    for _ in range(5):
        inputs = torch.randperm(P)[:int(torch.randint(1, P, ()))]
        rows = torch.cat((torch.randint(-1, C, (12,)), torch.randperm(C)[:5]))
        lanes = torch.rand(len(inputs), 3, 2, requires_grad=True)
        y = sigma(not_(lanes), inputs, rows)
        g = torch.randn_like(y)
        got = torch.autograd.grad(y, (sigma.value, lanes), g)
        y0 = _present_forward(sigma, NotLayer3.value(lanes), inputs, rows)
        want = torch.autograd.grad(y0, (sigma.value, lanes), g)
        torch.testing.assert_close(y, y0, atol=1e-6, rtol=0)
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
    print(dict(entries=sigma.nnz, not_entries=int(NotLayer3.entries(sigma).sum())))


# Rule 9, in part (certificate 6's first clause is not yet met: the store's
# columns are still the two lanes of each concept row, read as its symbol).

def test_a_concept_row_is_the_concepts_alone_and_a_symbol_has_no_address(lesson_at_start):
    owner, _ = lesson_at_start
    model = owner._model
    keys = list(owner._concept_allocator.layer(0)._tensor_rows)
    assert keys and all(isinstance(key, tuple) and key[0] == 'concept' for key in keys)
    assert not hasattr(owner, '_csw_row_of') and not hasattr(owner, '_csw_rows_of')
    assert 'symbols' not in model.where_registry.slices


def test_the_concept_a_row_brought_is_recorded_at_its_nomination(lesson_at_start):
    owner, _ = lesson_at_start
    brought = owner._concept_allocator.brought
    assert brought
    for percept, row in brought.items():
        assert owner.concept_of_basic_percept(percept) == row
        assert percept in owner.concept_percepts(row)


# The memberships of bytes in the properties are read by order zero and not
# taught by a concept's error (provisional; the question is with Alec).

def test_the_order_zero_word_read_passes_no_gradient_to_the_wholes_content():
    """The word step's order-zero evidence reads the wholes' content (the
    memberships read through the properties' codes) and gives it no
    gradient; the part event's gradient is untouched."""
    from types import SimpleNamespace
    from Spaces import ConceptualSpace
    owner = SimpleNamespace(outputShape=(3,), matched_code_evidence=ConceptualSpace.matched_code_evidence)
    parts = torch.rand(1, 3, 6, requires_grad=True)
    wholes = torch.rand(1, 3, 6, requires_grad=True)
    references = torch.rand(1, 2, 6)
    roles = torch.tensor([[0, 1]])
    evidence, mask = ConceptualSpace.percept_code_evidence(
        owner, parts, wholes, references, references, roles, torch.tensor([True]),
        part_n_what=4, whole_n_what=4)
    assert bool(mask[..., 1].any()) and bool(mask[..., 0].any())
    evidence.sum().backward()
    assert wholes.grad is None
    assert parts.grad is not None and bool(parts.grad.abs().gt(0).any())


def test_the_lessons_training_keeps_every_declared_membership_and_reads_no_known_word_both(capsys):
    """After the lesson's sixteen training steps (learning rate .001, no
    seed chosen): every membership declared at one is still exactly one,
    and no known word's against lane is above nothing (none of the
    lesson's words has a holder at its own extent, and its properties hold
    of all its bytes). The declared memberships at zero are printed with
    what moved them: the reconstruction of the form, through the
    properties' codes, which the cut does not touch."""
    from pytest import MonkeyPatch
    import ModelAttention
    import Language
    from Models import BaseModel
    from data import TheData
    from MathChainTraining import present
    from corpus_lessons import stage_documents
    from Corpus import admit_training_forms
    from reading_fixtures import use_eager_reading
    patch = MonkeyPatch()
    use_eager_reading(patch)
    against = []
    poles_of = ModelAttention.native_word_poles

    def recorded(model, spans, forms, known):
        poles = poles_of(model, spans, forms, known)
        against.extend(poles[..., 1][known.to(poles.device)].tolist())
        return poles
    patch.setattr(ModelAttention, 'native_word_poles', recorded)
    TheData.processLM({split: dict(text=['.'] if split == 'train' else [], label=[])
                      for split in ('train', 'validation', 'test')})
    Language.TheGrammar._configured = False
    model = BaseModel.from_config('data/MM_corpus_lessons.xml', data=TheData)[0]
    model._tensor_peer_while_eager = True
    optimizer = model.getOptimizer(lr=.001)
    curriculum = json.loads(Path('data/corpus-lessons.json').read_text())
    texts, targets = stage_documents(TheData, curriculum['stages'][0])
    admit_training_forms(model, texts, targets)
    properties = model.wholeSpace.subspace.what.primitive_properties
    declared = (properties.observed | properties.fixed_rows[:, None]).clone()
    start = properties.members.detach().clone()
    assert bool(((start[declared] == 0) | (start[declared] == 1)).all())
    count = 0

    class Complete(Exception):
        pass

    def batch(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 16:
            raise Complete
    try:
        present(model, split='train', optimizer=optimizer, batch_size=1, after_batch=batch)
    except Complete:
        pass
    finally:
        model.End()
        patch.undo()
    after = properties.members.detach()
    ones, zeros = declared & (start == 1), declared & (start == 0)
    moved_zeros = zeros & (after != 0)
    with capsys.disabled():
        print(f'\ndeclared memberships: {int(ones.sum())} at one, {int(zeros.sum())} at zero; '
              f'at zero and moved: {int(moved_zeros.sum())} in rows '
              f'{sorted(set(moved_zeros.nonzero()[:, 0].tolist()))}, greatest '
              f'{float(after[zeros].max()) if bool(zeros.any()) else 0.:.3g}; '
              f'known words read: {len(against)}, greatest against {max(against, default=0.):.3g}')
    assert count == 16 and against
    assert torch.equal(after[ones], start[ones])
    assert max(against) == 0.


# Certificate 4, on the making of symbols: a symbol is made at MTM's
# nomination above order zero and brings one concept; a concept's
# allocation makes none. Its two lanes are a row of the symbols' Pi.

def test_4_a_concepts_allocation_makes_no_symbol():
    from test_cs_sparse_weights import _cs
    cs = _cs(nS=16, order=2)
    table = cs._concept_allocator.symbol_table()
    assert len(table) == 0
    made = [cs.new_concept(), cs.new_concept(order=1), cs.new_concept(order=2)]
    assert len(set(made)) == 3 and len(table) == 0


def test_4_a_nomination_above_order_zero_makes_one_symbol_that_brings_its_concept(lesson_at_start):
    owner, _ = lesson_at_start
    alloc = owner._concept_allocator
    table = alloc.symbol_table()
    print(f'\nsymbols made by nomination in the lesson: {len(table)}; '
          f'handles and the concepts they brought: {list(zip(table.handles, table.brought))}')
    assert len(table) > 0
    for symbol, (handle, concept) in enumerate(zip(table.handles, table.brought)):
        assert table.of(handle) == symbol
        # MTM's nominations bring a concept above order zero; a symbol made
        # for a kind may hold a member of any lower order.
        assert handle[0] == 'kind' or alloc.order_of(concept) >= 1
        # It starts with one member, the concept it brought.
        assert table.members(symbol) == [concept]
        entry = owner.mtm.entries.get(handle)
        assert entry is None or entry.symbol == concept
    # The Pi gives each symbol its concept's two lanes: one member at one.
    lanes = torch.rand(len(table), 3, 2)
    torch.testing.assert_close(table.read(lanes, table.brought), lanes, rtol=0, atol=1e-6)


def test_6_a_symbol_has_its_own_index_and_no_vector_and_nothing_is_one_to_one():
    """No tensor of the symbol table holds a concept: its Pi's rows are
    symbols, addressed by the table's own index, and their members are
    keys of concepts; a symbol holds two lanes and no code. A symbol may
    read several concepts and a concept may be read by several symbols."""
    from Symbols import SymbolTable
    table = SymbolTable(8)
    first = table.make(('name', 40), 40)
    second = table.make(('candidate', 3), 40)
    assert (first, second) == (0, 1) and table.make(('name', 40), 41) == first
    assert table.reading(40) == [0, 1] and table.brought == [40, 40]
    restored = SymbolTable(8)
    state = table.state()
    state['symbols'].append((('candidate', 4), 41, (41, 40)))
    restored.restore(state)
    assert restored.members(2) == [41, 40] and restored.reading(40) == [0, 1, 2]
    lanes = torch.tensor([[[.5, 0.]], [[.8, .1]]])
    read = restored.read(lanes, [40, 41])
    torch.testing.assert_close(read[2, 0], torch.tensor([.4, 0.]), rtol=0, atol=1e-6)
    # Nothing addresses "the" symbol of a concept: the fit is the symbol
    # with the fewest constituents among those that hold it.
    assert restored.fit(41) == 2 and restored.fit(40) == 0 and restored.fit(39) is None
    # The Pi's storage is by symbol, and its member keys name concepts.
    assert len(restored.pi.crow) == 8 + 1
    assert all(key[0] == 'concept' for key in restored.pi.keys)


# Rule 71: no symbol is made for a concept. A symbol is a row whose
# constituents occur together, made when MTM nominates it.

def _calls_in_bin(name):
    """Every call of ``name`` (as a function or a method) in bin/, as
    (file, enclosing function)."""
    import ast
    found = []
    for path in sorted(Path('bin').glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        for scope in ast.walk(tree):
            if not isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
                continue
            for node in ast.walk(scope):
                if isinstance(node, ast.Call):
                    func = node.func
                    called = func.attr if isinstance(func, ast.Attribute) else getattr(func, 'id', None)
                    if called == name and scope is not tree:
                        found.append((path.name, scope.name))
    return sorted(set(found))


def test_71_no_symbol_is_made_for_a_concept_and_every_symbol_is_mtms(lesson_at_start):
    import ast
    defined = set()
    handles = []
    for path in sorted(Path('bin').glob('*.py')):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                defined.add(node.name)
            # A handle written as a tuple whose first item names a making
            # for a concept, passed to a maker of symbols.
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ('nominate_symbol', 'make') and node.args
                    and isinstance(node.args[0], ast.Tuple) and node.args[0].elts
                    and isinstance(node.args[0].elts[0], ast.Constant)
                    and node.args[0].elts[0].value in ('declared', 'percept', 'object', 'part-kind')):
                handles.append((path.name, node.lineno))
    # No function gives "the" symbol of a concept or makes one for it.
    assert not defined & {'symbol_for', 'symbols_from_records'}
    assert not handles
    # The one maker is called by MTM's two nominations and by a kind's need
    # of a member that no symbol holds, and by nothing else.
    assert _calls_in_bin('nominate_symbol') == [
        ('FamiliarTier.py', 'promote_candidates'), ('FamiliarTier.py', 'promote_identity'),
        ('Spaces.py', 'things_reading')]
    # After the lesson's sixteen sentences, every symbol in the table was
    # nominated by MTM (its handle is an entry of MTM whose symbol is the
    # concept the nomination brought), or made for a kind, named by the
    # making ('kind', the kind, the member) and read by that kind.
    owner, _ = lesson_at_start
    table = owner._concept_allocator.symbol_table()
    store = owner._concept_allocator.layer(0)
    print(f'\nsymbols after the lesson: {len(table)}, handles {list(table.handles)}')
    assert len(table) > 0
    for symbol, (handle, concept) in enumerate(zip(table.handles, table.brought)):
        entry = owner.mtm.entries.get(handle)
        if handle[0] == 'kind':
            assert handle[2] == concept and table.members(symbol) == [concept]
            assert (owner._concept_row_of(handle[1]), store.column_of_thing(('sym', symbol))) in store._index
        else:
            assert entry is not None and entry.symbol == concept, handle


def test_71_a_checkpoint_drops_the_symbols_made_for_a_concept():
    """A checkpoint saved while symbols were made for concepts holds them
    under the handles of those makings; reading it drops them, by handle,
    with a warning, and keeps MTM's nominations."""
    from Models import _symbols_nominated_by_mtm
    from Symbols import SymbolTable
    saved = SymbolTable(8)
    saved.make(('declared', 'formal-argument', 'agent'), 3)
    saved.make(('name', 40), 40)
    saved.make(('percept', 7), 5)
    saved.make(('object', 9), 9)
    saved.make(('part-kind', (1, 4), 2), 12)
    saved.make(('candidate', 3), 41)
    with pytest.warns(UserWarning, match='4 symbols made for a concept'):
        state = _symbols_nominated_by_mtm(saved.state())
    table = SymbolTable(8)
    table.restore(state)
    assert table.handles == [('name', 40), ('candidate', 3)]
    assert table.brought == [40, 41] and table.of(('candidate', 3)) == 1
    assert table.members(1) == [41]
    # A checkpoint of MTM's nominations alone is read as it is.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        assert _symbols_nominated_by_mtm(table.state()) == table.state()


# Rule 71: what a kind reads is a row of a Pi, a thing. The store's columns
# are the basic percepts' rows and the symbols' rows, with the bias.

def _store_things(owner):
    """Every entry of the store as (reader row, thing, lane)."""
    store = owner._concept_allocator.layer(0)
    return [(row, store.thing_of_column(column), store.column_lane(column))
            for row, column in zip(store._rows, store._cols)]


def test_cert_a_no_column_of_the_store_is_addressed_by_a_concepts_row(lesson_at_start):
    """(a) No reader in bin/ decodes a column as a concept's row (no column
    arithmetic over the store's row count), and after the lesson every
    entry of the store reads a thing: a nominated basic percept or a symbol
    of the table, or the bias."""
    import re
    pattern = re.compile(r'nOutput\s*\+\s*1|\(\s*n\s*\+\s*1\s*\)|_bias\s*\+\s*1')
    readers = ('Layers.py', 'Spaces.py', 'FieldProjection.py', 'CaseSelection.py', 'DefinitionFold.py')
    found = [(name, number, line.strip()) for name in readers
             for number, line in enumerate(Path('bin', name).read_text().splitlines(), 1)
             if pattern.search(line) and 'old_size' not in line]
    assert not found, found
    owner, _ = lesson_at_start
    store = owner._concept_allocator.layer(0)
    basic, table = owner.basic_percepts, store.symbol_table
    entries = _store_things(owner)
    kinds = {'bp': 0, 'sym': 0, 'bias': 0}
    for row, thing, lane in entries:
        assert lane in (0, 1)
        if thing is None:
            kinds['bias'] += 1
        elif thing[0] == 'bp':
            kinds['bp'] += 1
            assert bool(basic.promoted[thing[1]]), thing
        else:
            kinds['sym'] += 1
            assert thing[1] < len(table), thing
    print(f'\nthe lesson store: {len(entries)} entries; on basic percepts {kinds["bp"]}, '
          f'on symbols {kinds["sym"]}, on the bias {kinds["bias"]}; symbols in the table {len(table)}')
    assert kinds['bp'] and len(entries) == sum(kinds.values())


def test_cert_b_words_reads_a_word_on_its_basic_percept(lesson_at_start):
    """(b) The kind `words` reads a word on the word's basic percept: where
    the basic percept is present it reads it, where it is not it does not,
    with no symbol and no order-zero concept between (none is seated)."""
    from dataclasses import replace
    owner, fields = lesson_at_start
    alloc = owner._concept_allocator
    words_row = owner._concept_row_of(alloc.declared_classes['words'])
    reads = [thing for row, thing, lane in _store_things(owner) if row == words_row and lane == 0]
    assert reads and all(thing[0] == 'bp' for thing in reads), reads
    percept = reads[0][1]
    perception = next(f for f in fields if ('bp', percept) in f.keys)
    index = perception.keys.index(('bp', percept))
    present = torch.zeros_like(perception.values)
    present[index, ..., 0] = 1.
    rows = torch.tensor([words_row])
    seats = torch.zeros(1, *perception.values.shape[1:3], 2)
    read = lambda values: owner._concept_rung_presence(
        seats, 0, 1, inventory_rows=rows, percepts=replace(perception, values=values))
    lit, unlit = read(present), read(torch.zeros_like(present))
    print(f'\nwords on basic percept {percept}: present {lit[0, 0, 0].tolist()}, absent {unlit[0, 0, 0].tolist()}')
    assert bool((lit[..., 0] > .5).all()) and bool((unlit[..., 0] == 0).all())
    assert not any(owner.concept_id_at_row(r) is not None and owner._row_order(r) == 0 for r in rows.tolist())


def test_cert_c_an_operator_class_reads_its_arguments_through_symbols(lesson_at_start):
    """(c) An operator class reads its arguments through symbols, made by
    the class's taking of its argument; and the table holds no symbol that
    no kind reads and that MTM did not nominate."""
    owner, _ = lesson_at_start
    alloc = owner._concept_allocator
    table = alloc.symbol_table()
    arguments = set(alloc.formal_arguments.values())
    classes = {owner._concept_row_of(c) for c in alloc.operator_concepts.values()}
    entries = _store_things(owner)
    read_symbols = {thing[1] for _, thing, _ in entries if thing is not None and thing[0] == 'sym'}
    by_class = [(row, thing) for row, thing, lane in entries if row in classes and thing is not None]
    assert by_class
    for row, thing in by_class:
        assert thing[0] == 'sym'
        members = table.members(thing[1])
        assert set(members) <= arguments, members
        handle = table.handles[thing[1]]
        assert handle[0] in ('kind', 'name', 'candidate'), handle
    for symbol, handle in enumerate(table.handles):
        entry = owner.mtm.entries.get(handle)
        assert symbol in read_symbols or (entry is not None and entry.symbol == table.brought[symbol]), handle
    print(f'\noperator classes read {len(by_class)} symbols over arguments; table {len(table)}, '
          f'read by a kind {len(read_symbols)}')


def _small_space():
    """A space whose store has room for 8 basic percepts; concepts at
    orders zero, one and two."""
    from test_cs_sparse_weights import _cs, _mint_row
    cs = _cs(nS=16, order=3)
    store = cs._concept_allocator.layer(0)
    store.set_room(percepts=8)
    ids = [cs.new_concept() for _ in range(5)]
    rows = [_mint_row(cs, order, cid) for order, cid in zip((0, 1, 1, 2, 2), ids)]
    return cs, store, ids, rows


def test_cert_d_green_things_collect_from_every_lower_order():
    """(d) One kind of order two reads a basic percept (Felix's word) and a
    symbol over a concept of order one (cats); either evidenced is enough.
    An entry on a thing of its own order is refused, with the reason."""
    from dataclasses import dataclass
    cs, store, ids, rows = _small_space()
    word, cats, _, green, other = rows
    cs._concept_allocator.brought[3] = word
    cs.add_concept_edge(green, word, 1.)
    cs.add_concept_edge(green, cats, 1.)
    things = {thing for row, thing, _ in _store_things(cs) if row == green}
    symbol = store.symbol_table.fit(ids[1])
    assert things == {('bp', 3), ('sym', symbol)}
    assert store.symbol_table.handles[symbol] == ('kind', ids[3], ids[1])

    @dataclass
    class Reading:
        keys: tuple
        values: torch.Tensor
    inventory = torch.tensor([cats, green])

    def read(felix, cat):
        seats = torch.tensor([[[[cat, 0.]]], [[[0., 0.]]]])
        values = torch.tensor([[[[felix, 0.]]]])
        return cs._concept_rung_presence(seats, 1, 2, inventory_rows=inventory,
            percepts=Reading((('bp', 3),), values))[0, 0, 0, 0]
    assert float(read(1., 0.)) > .5 and float(read(0., .8)) > .5 and float(read(0., 0.)) == 0.
    with pytest.raises(ValueError, match='lower orders only'):
        cs.add_concept_edge(green, other, 1.)
    with pytest.raises(ValueError, match='lower orders only'):
        cs.add_thing_edge(green, ('sym', cs.nominate_symbol(('test', ids[4]), ids[4])), 1.)


def test_cert_e_a_symbol_of_two_constituents_reads_their_product_once():
    """(e) A symbol of two constituents reads their product on each lane,
    by the same class and fold as a basic percept (SparsePiLayer3); a kind
    that reads the symbol reads it once."""
    from OrderZero import SparsePiLayer3
    cs, store, ids, rows = _small_space()
    table = store.symbol_table
    assert isinstance(table.pi, SparsePiLayer3)
    state = table.state()
    state['symbols'].append((('test', 'two'), ids[1], (ids[1], ids[2])))
    table.restore(state)
    symbol = table.of(('test', 'two'))
    kind = rows[3]
    cs.add_thing_edge(kind, ('sym', symbol), 1.)
    assert [thing for row, thing, _ in _store_things(cs) if row == kind] == [('sym', symbol)]
    lanes = torch.tensor([[[.5, .2]], [[.8, .5]]])
    torch.testing.assert_close(table.read(lanes, ids[1:3], [symbol])[0, 0], torch.tensor([.4, .1]))
    view = store.bind(torch.tensor(rows[1:4]))
    from Layers import thing_lanes
    seats = torch.tensor([[[.5, .2]], [[.8, .5]], [[0., 0.]]])
    torch.testing.assert_close(thing_lanes(view, seats)[0, 0], torch.tensor([.4, .1]))
    assert len(view._things) == 1


def test_cert_f_nothing_is_one_to_one():
    """(f) A thing read by two kinds, a kind reading many things, a concept
    in two symbols."""
    cs, store, ids, rows = _small_space()
    word, cats, dogs, green, small = rows
    cs._concept_allocator.brought[3] = word
    cs._concept_allocator.brought[4] = word
    for kind in (green, small):
        cs.add_concept_edge(kind, cats, 1.)
    cs.add_concept_edge(green, dogs, 1.)
    cs.add_thing_edge(green, ('bp', 5), 1.)
    entries = _store_things(cs)
    symbol = store.symbol_table.fit(ids[1])
    assert {row for row, thing, _ in entries if thing == ('sym', symbol)} == {green, small}
    assert len({thing for row, thing, _ in entries if row == green}) == 3
    second = cs.nominate_symbol(('test', 'second'), ids[1])
    assert second != symbol and set(store.symbol_table.reading(ids[1])) == {symbol, second}
    # A concept two basic percepts brought is read on both.
    assert set(cs.things_reading(green, word)) == {('bp', 3), ('bp', 4)}


def test_cert_8_an_old_store_whose_columns_were_concept_rows_is_read_once_as_things():
    """An old checkpoint's store wrote each column as a concept's row. Read
    once: a column of a concept that a basic percept's nomination brought
    becomes that basic percept's row; any other becomes a symbol found or
    made for the kind that read it; the bias stays the bias; values kept."""
    import torch.nn as nn
    from Models import _read_rows_as_things
    cs, store, ids, rows = _small_space()
    word, cats, _, green, _ = rows
    cs._concept_allocator.brought[6] = word
    # Three entries of `green`, written the old way: (source row, lane).
    read_as_rows = [(word, 0), (cats, 1), (None, 0)]
    store._rows, store._cols = [green] * 3, [0, 1, 2]
    store._init_vals = [.5, .25, 1.]
    store._index = {(green, c): i for i, c in enumerate(store._cols)}
    store.values = nn.Parameter(torch.tensor([.5, .25, 1.]))
    _read_rows_as_things(cs, store, read_as_rows)
    things = [(store.thing_of_column(c), store.column_lane(c)) for c in store._cols]
    symbol = store.symbol_table.fit(ids[1])
    assert things == [(('bp', 6), 0), (('sym', symbol), 1), (None, 0)]
    assert store.symbol_table.handles[symbol] == ('kind', ids[3], ids[1])
    torch.testing.assert_close(store.values.detach(), torch.tensor([.5, .25, 1.]))
