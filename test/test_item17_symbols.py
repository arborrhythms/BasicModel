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
import math
from pathlib import Path

import pytest
import torch

from Layers import NotLayer3, SparseSigmaLDU
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
    not_, sigma = NotLayer3(rows), SparseSigmaLDU(rows, concepts, 32)
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
    not_, sigma = NotLayer3(2), SparseSigmaLDU(2, 3, 8)
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
    not_, sigma = NotLayer3(1), SparseSigmaLDU(1, 1, 1)
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
    sigma = SparseSigmaLDU(4, 2, 8)
    saved = {'weight': torch.rand(6, 2), 'against': torch.rand(6, 2)}
    with pytest.raises(ValueError, match='no reading on the rows of the first Pi'):
        sigma.load_state_dict(saved, strict=False)


# Certificate 54: the Sigma of order zero reads rows (rule 62).

ORDER_ZERO_READERS = {
    'Layers.py': ('NotLayer3', 'SparseLDULayer', 'SparseSigmaLDU'),
    'OrderZero.py': ('read_word',),
    'NumericPercepts.py': ('read',),
    'Spaces.py': ('order_zero_reading', 'install_order_zero', 'order_zero_field', '_start_order_zero_entry',
                  'cs_read_percept_field', 'add_concept_feature'),
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
    On ``reach``, which does not hold the part of its own that ``archer``
    has as a stored form, it reads zero.
    Read through the first Pi, the Not and the Sigma of order zero's
    classes, one concept on each row, at its start."""
    from test_item17_code import coded, _read_form
    words = (b'archer', b'archery', b'is', b'this', b'reach')
    identity, basic, book, rows = coded(words)
    not_ = NotLayer3(basic.capacity)
    sigma = SparseSigmaLDU(basic.capacity, len(words), 2 * len(words))
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
    what arrives at a look, which the concepts' stage of expectation is
    compared with (``ConceptSpread.look_cost``), gives the words of the
    sentence read and zero to the words of the other."""
    import ConceptSpread
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
        taught = {}
        cost = ConceptSpread.look_cost

        def kept_cost(owner, before, arrived):
            taught.setdefault(model._reading_text, []).append(arrived[0, concepts, 0].tolist())
            return cost(owner, before, arrived)
        monkeypatch.setattr(ConceptSpread, 'look_cost', kept_cost)
        omit_second(monkeypatch, model)
        optimizer = model.getOptimizer(lr=.01)
        for text in examples:
            model._reading_text = text
            read(model, text, training=True, optimizer=optimizer)
        print(dict(concepts=concepts, readings=table, teacher=taught))
        assert set(taught) == set(examples)
        for k, text in enumerate(examples):
            values = torch.tensor(taught[text]).amax(0).tolist()
            mine, other = (values[:2], values[2:]) if k == 0 else (values[2:], values[:2])
            assert max(mine) > .5 and max(other) == 0.
    finally:
        model.End(); model.symbolSpace.soft_reset()


# The sparse Sigma's read costs the entries it reads (rule 19): the same
# values and gradients as the read it replaced, which passed over every
# written entry with an index as long as the inputs' capacity.

def _present_forward(sigma, signed, inputs, rows):
    """The read over every written entry: ``tanh`` of each concept's column
    of W times its rows' signed values. Ported (eighth pass): it read each
    entry's learned number as W's entry (the read of 1f58fe1511); the learned numbers are the factors now, so W is
    the dense ``L D U`` of them (``sparse_ldu.dense_w``), products
    included, differentiable in the factors."""
    from sparse_ldu import dense_w
    inputs = torch.as_tensor(inputs, dtype=torch.long).reshape(-1)
    rows = torch.as_tensor(rows, dtype=torch.long).reshape(-1)
    shape = (len(rows), *signed.shape[1:])
    if not sigma.nnz or not len(rows) or not len(inputs):
        return signed.new_zeros(shape)
    w = dense_w(sigma).to(signed)
    known = (inputs >= 0) & (inputs < sigma.nInput)
    x = signed.new_zeros(sigma.nInput, *signed.shape[1:]).index_add(0, inputs[known], signed[known])
    y = torch.tanh(torch.einsum('i...,ic->c...', x, w))
    valid = (rows >= 0) & (rows < sigma.nOutput)
    y = y.index_select(0, rows.clamp(0, sigma.nOutput - 1))
    return y * valid.reshape(-1, *([1] * (y.ndim - 1))).to(y)


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
    not_, sigma = NotLayer3(P), SparseSigmaLDU(P, C, 600)
    for _ in range(500):
        c, r = int(torch.randint(C, ())), int(torch.randint(P, ()))
        if float(torch.rand(())) < nots:
            not_.make(sigma, c, r, float(torch.rand(())))
        else:
            sigma.add(c, r, float(torch.rand(())) * (float(torch.rand(())) > .1))
    # In 64 bits: the factors' gradients carry products of entries, some
    # in the tens, which the dense read and the scatters round apart in 32.
    sigma.double()
    for _ in range(5):
        inputs = torch.randperm(P)[:int(torch.randint(1, P, ()))]
        rows = torch.cat((torch.randint(-1, C, (12,)), torch.randperm(C)[:5]))
        lanes = torch.rand(len(inputs), 3, 2, dtype=torch.float64, requires_grad=True)
        y = sigma(not_(lanes), inputs, rows)
        g = torch.randn_like(y)
        got = torch.autograd.grad(y, (sigma.value, lanes), g)
        y0 = _present_forward(sigma, NotLayer3.value(lanes), inputs, rows)
        want = torch.autograd.grad(y0, (sigma.value, lanes), g)
        torch.testing.assert_close(y, y0, atol=1e-6, rtol=0)
        for a, b in zip(got, want):
            torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
    print(dict(entries=sigma.nnz, not_entries=int(NotLayer3.entries(sigma).sum()),
               lower=sigma.lower_entries, chains=(sigma.longest_chain, sigma.longest_chain_lower)))


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


# The memberships of bytes in the properties are read by order zero and
# taught by a concept's error: declared classes are not fixed.

def test_the_order_zero_word_read_passes_its_gradient_to_the_wholes_content():
    """The word step's order-zero evidence reads the wholes' content (the
    memberships read through the properties' codes) and gives it its
    gradient, as it gives the part event its own."""
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
    assert wholes.grad is not None and bool(wholes.grad.abs().gt(0).any())
    assert parts.grad is not None and bool(parts.grad.abs().gt(0).any())


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
    # A making finds the symbol that already holds its concepts first; a
    # second symbol over 40 is one that holds another concept too.
    assert table.make(('candidate', 2), 40) == first
    second = table.make(('candidate', 3), 40, 41)
    assert (first, second) == (0, 1) and table.make(('name', 40), 41) == first
    assert table.reading(40) == [0, 1] and table.brought == [40, 40]
    restored = SymbolTable(8)
    state = table.state()
    state['symbols'].append((('candidate', 4), 41, (41, 40)))
    restored.restore(state)
    assert restored.members(2) == [41, 40] and restored.reading(40) == [0, 1, 2]
    lanes = torch.tensor([[[.5, 0.]], [[.8, .1]]])
    read = restored.read(lanes, [40, 41])
    torch.testing.assert_close(read[1, 0], torch.tensor([.4, 0.]), rtol=0, atol=1e-6)
    torch.testing.assert_close(read[2, 0], torch.tensor([.4, 0.]), rtol=0, atol=1e-6)
    # Nothing addresses "the" symbol of a concept: the fit is the symbol
    # with the fewest constituents among those that hold it.
    assert restored.fit(41) == 1 and restored.fit(40) == 0 and restored.fit(39) is None
    # The Pi's storage is by symbol, and its members name concepts. Ported
    # (eighth pass): the compressed rows of SparsePiLayer3 (``crow`` of 8 + 1,
    # member keys ``('concept', cid)``) are the outputs and inputs of the LDU.
    assert restored.pi.nOutput == 8
    assert restored.concepts == [40, 41]
    # The way back (rule 78): each concept from the symbol it brought.
    back = restored.back(restored.read(lanes, [40, 41]), [40, 41])
    print(dict(back=back.tolist()))


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
        if handle[0] == 'same':
            # MTM's making from a statement that two are the same.
            assert sorted(table.members(symbol)) == sorted(handle[1:]) and concept == handle[1]
        elif handle[0] == 'kind':
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



# Rule 71: what a kind reads is a row of a Pi, a thing: above order zero, a
# symbol. The store's columns are the symbols' rows, with the bias.

def _store_things(owner):
    """Every entry of the store as (reader row, thing, Not entry)."""
    store = owner._concept_allocator.layer(0)
    return [(row, store.thing_of_column(column), store.negated(position))
            for position, (row, column) in enumerate(zip(store._rows, store._cols))]


def test_cert_a_no_column_of_the_store_is_addressed_by_a_concepts_row(lesson_at_start):
    """(a) No reader in bin/ decodes a column as a concept's row (no column
    arithmetic over the store's row count), and after the lesson every
    entry of the store reads a symbol of the table, or the bias; none a
    basic percept."""
    import re
    pattern = re.compile(r'nOutput\s*\+\s*1|\(\s*n\s*\+\s*1\s*\)|_bias\s*\+\s*1')
    readers = ('Layers.py', 'Spaces.py', 'FieldProjection.py', 'CaseSelection.py', 'DefinitionFold.py')
    found = [(name, number, line.strip()) for name in readers
             for number, line in enumerate(Path('bin', name).read_text().splitlines(), 1)
             if pattern.search(line)]
    assert not found, found
    owner, _ = lesson_at_start
    store = owner._concept_allocator.layer(0)
    table = store.symbol_table
    entries = _store_things(owner)
    on_symbols = [thing for _, thing, _ in entries if thing is not None]
    assert all(thing[0] == 'sym' and thing[1] < len(table) for thing in on_symbols)
    assert store.nInput == table.capacity + 1
    print(f'\nthe lesson store: {len(entries)} entries, {len(on_symbols)} on symbols, '
          f'{len(entries) - len(on_symbols)} on the bias; symbols in the table {len(table)}')
    assert on_symbols


def test_cert_b_words_reads_a_word_through_a_symbol_over_its_concept(lesson_at_start):
    """(b) The kind `words` reads a word through a symbol over the word's
    order-zero concept: where the concept is present it reads it, where it
    is not it does not. No column is a basic percept."""
    owner, _ = lesson_at_start
    alloc = owner._concept_allocator
    table = alloc.symbol_table()
    words_row = owner._concept_row_of(alloc.declared_classes['words'])
    reads = [thing for row, thing, negated in _store_things(owner) if row == words_row and not negated]
    assert reads and all(thing[0] == 'sym' for thing in reads), reads
    member, = table.members(reads[0][1])
    word_row = owner._concept_row_of(member)
    assert owner._row_order(word_row) == 0
    rows = torch.tensor([word_row, words_row])
    def read(value):
        seats = torch.tensor([[[[value, 0.]]], [[[0., 0.]]]])
        return owner._concept_rung_presence(seats, 1, 2, inventory_rows=rows)[0, 0, 0]
    lit, unlit = read(.76), read(0.)
    print(f'\nwords through symbol {reads[0][1]} over word concept {member}: '
          f'present {lit.tolist()}, absent {unlit.tolist()}')
    # The Sigma's read: tanh of the weight (one) times the symbol's value.
    assert float(lit[0]) == pytest.approx(math.tanh(.76)) and float(lit[1]) == 0.
    assert bool((unlit == 0).all())


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
    read_symbols = {thing[1] for _, thing, _ in entries if thing is not None}
    by_class = [(row, thing) for row, thing, lane in entries if row in classes and thing is not None]
    assert by_class
    for row, thing in by_class:
        members = table.members(thing[1])
        assert set(members) <= arguments, members
        handle = table.handles[thing[1]]
        assert handle[0] in ('kind', 'name', 'candidate'), handle
    for symbol, handle in enumerate(table.handles):
        entry = owner.mtm.entries.get(handle)
        assert symbol in read_symbols or (entry is not None and entry.symbol == table.brought[symbol]), handle
    print(f'\noperator classes read {len(by_class)} entries on symbols over arguments; '
          f'table {len(table)}, read by a kind {len(read_symbols)}')


def _small_space():
    """A space with concepts at orders zero, one, one, two and two."""
    from test_cs_sparse_weights import _cs, _mint_row
    cs = _cs(nS=16, order=3)
    store = cs._concept_allocator.layer(0)
    ids = [cs.new_concept() for _ in range(5)]
    rows = [_mint_row(cs, order, cid) for order, cid in zip((0, 1, 1, 2, 2), ids)]
    return cs, store, ids, rows


def test_cert_d_green_things_collect_from_every_lower_epistemic_level():
    """(d) One kind of order two reads a symbol over an order-zero concept
    (Felix) and a symbol over an order-one concept (cats); either evidenced
    is enough. A symbol with constituents of two orders is refused, and so
    is an entry on a symbol at or above the kind's own epistemic level."""
    cs, store, ids, rows = _small_space()
    felix, cats, _, green, other = rows
    cs.add_concept_edge(green, felix, 1.)
    cs.add_concept_edge(green, cats, 1.)
    table = store.symbol_table
    things = {thing for row, thing, _ in _store_things(cs) if row == green}
    assert things == {('sym', table.fit(ids[0])), ('sym', table.fit(ids[1]))}
    assert table.handles[table.fit(ids[1])] == ('kind', ids[3], ids[1])
    inventory = torch.tensor([felix, cats, green])

    def read(f, c):
        seats = torch.tensor([[[[f, 0.]]], [[[c, 0.]]], [[[0., 0.]]]])
        return float(cs._concept_rung_presence(seats, 2, 3, inventory_rows=inventory)[0, 0, 0, 0])
    assert read(.8, 0.) > .5 and read(0., .8) > .5 and read(0., 0.) == 0.
    # A list: the two members' values are summed.
    assert read(.8, .8) == pytest.approx(math.tanh(1.6))
    with pytest.raises(ValueError, match='lower epistemic levels only'):
        cs.add_concept_edge(green, other, 1.)
    state = table.state()
    state['symbols'].append((('test', 'mixed'), ids[0], (ids[0], ids[1])))
    table.restore(state)
    with pytest.raises(ValueError, match='concepts of one order'):
        cs.add_thing_edge(green, ('sym', table.of(('test', 'mixed'))), 1.)
    assert len(table) == 3


def test_cert_e_a_symbol_of_two_constituents_reads_their_product_once():
    """(e) A symbol of two constituents reads their product on each lane,
    by the Pi's fold, the product (the symbols' Pi is a SparsePiLDU since
    the eighth pass, ported from SparsePiLayer3, the basic percepts'
    class); a kind that reads the symbol reads it once."""
    from Layers import SparsePiLDU, thing_lanes
    cs, store, ids, rows = _small_space()
    table = store.symbol_table
    assert isinstance(table.pi, SparsePiLDU)
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
    seats = torch.tensor([[[.5, .2]], [[.8, .5]], [[0., 0.]]])
    torch.testing.assert_close(thing_lanes(view, seats)[0, 0], torch.tensor([.4, .1]))
    assert len(view._things) == 1


def test_cert_f_nothing_is_one_to_one():
    """(f) A symbol read by two kinds, a kind reading many symbols, a
    concept in two symbols."""
    cs, store, ids, rows = _small_space()
    word, cats, dogs, green, small = rows
    for kind in (green, small):
        cs.add_concept_edge(kind, cats, 1.)
    cs.add_concept_edge(green, dogs, 1.)
    cs.add_concept_edge(green, word, 1.)
    entries = _store_things(cs)
    symbol = store.symbol_table.fit(ids[1])
    assert {row for row, thing, _ in entries if thing == ('sym', symbol)} == {green, small}
    assert len({thing for row, thing, _ in entries if row == green}) == 3
    # A making finds the symbol that holds its concept first; a concept is
    # in a second symbol that holds another concept with it.
    assert cs.nominate_symbol(('test', 'again'), ids[1]) == symbol
    second = store.symbol_table.make(('test', 'second'), ids[1], ids[2])
    assert second != symbol and set(store.symbol_table.reading(ids[1])) == {symbol, second}


def test_cert_8_an_old_store_is_refused_with_the_reason():
    """A checkpoint whose store joined its inputs by the least and the
    greatest (a column for each lane of each input, unmarked) is refused,
    and the refusal says why."""
    from types import SimpleNamespace
    from Models import BaseModel
    cs, store, ids, rows = _small_space()
    old = {'next_id': 100, 'layers': {0: {
        'nInput': 2 * (store._room + 1), 'nOutput': store.nOutput, 'constituents': {},
        'tensor_rows': {}, 'row_next': {}, 'rows': [rows[3]], 'cols': [0], 'columns': 'symbols',
        'symbols': store._room, 'init_vals': [.5], 'values': torch.tensor([.5])}}}
    with pytest.raises(ValueError, match='the least and the greatest'):
        BaseModel._restore_allocator_extras(SimpleNamespace(), cs, old)


# Certificates 4, 13 and 23 (d) above order zero: the symbols' Pi gives
# each symbol two lanes, the Not its value and not-value, and the store's
# Sigma reads one option for each entry, with weights at zero or above.

def _member_lanes(cs, rows, member, lanes):
    seats = torch.zeros(len(rows), 1, 1, 2)
    seats[rows.index(member), 0, 0] = torch.tensor(lanes)
    return seats


def test_cert_4_the_next_sigma_reads_one_combined_input_for_each_symbol():
    """Certificate 4: the kind reads one input for its symbol, for less
    against (.7 - .2), by its weight, through tanh; the bound view has one
    column for each symbol and the bias."""
    cs, store, ids, rows = _small_space()
    word, cats, dogs, green, _ = rows
    cs.add_concept_edge(cats, word, 2.)
    inventory = [word, cats]
    view = store.bind(torch.tensor(inventory))
    assert view.nInput == len(view._things) + 1 == 2
    value = cs._concept_rung_presence(_member_lanes(cs, inventory, word, (.7, .2)), 1, 2,
                                      inventory_rows=torch.tensor(inventory))[0, 0, 0]
    print(f'\ncertificate 4: member (.7, .2), weight 2: kind {value.tolist()}')
    torch.testing.assert_close(value, torch.tensor([math.tanh(2 * .5), 0.]))
    assert all(store.symbol_table.members(t[1]) for t in view._things)


def test_cert_23d_above_order_zero_a_not_entry_reads_a_symbols_pair_exchanged():
    """Certificate 23 (d): a Not entry above order zero is an entry whose
    sign chooses the symbol's not-value: its weight is at zero or above
    (the entry's size), the kind reads the pair exchanged, and a kind of
    presence and one of absence over one symbol are each other's
    opposite."""
    cs, store, ids, rows = _small_space()
    word, cats, dogs, green, _ = rows
    cs.add_concept_edge(cats, word, 1., negated=True)
    cs.add_concept_edge(dogs, word, 1.)
    assert cs.concept_weights(cats, negated=True) == [(word, 1.)]
    assert cs.concept_weights(cats) == [] and cs.concept_weights(dogs) == [(word, 1.)]
    assert float(store.values.detach().abs().min()) >= 0 and store.negated(cs.concept_entries(cats, word)[0])
    inventory = [word, cats, dogs]
    read = cs._concept_rung_presence(_member_lanes(cs, inventory, word, (.7, .2)), 1, 3,
                                     inventory_rows=torch.tensor(inventory))[:, 0, 0]
    print(f'\ncertificate 23 (d): member (.7, .2): absence {read[0].tolist()}, presence {read[1].tolist()}')
    torch.testing.assert_close(read[0], torch.tensor([0., math.tanh(.5)]))
    torch.testing.assert_close(read[0], read[1].flip(-1))


def test_cert_13_an_entry_above_order_zero_that_no_cost_supports_goes_to_zero():
    """Certificate 13: under the penalty alone (no other cost), a Sigma
    entry above order zero falls to zero, whichever option it reads; the
    order-zero Sigma is not in the penalty."""
    cs, store, ids, rows = _small_space()
    word, cats, dogs, green, _ = rows
    cs.add_concept_edge(cats, word, .3)
    cs.add_concept_edge(dogs, word, .3, negated=True)
    optimizer = torch.optim.SGD([store.values], lr=.02)
    for _ in range(30):
        optimizer.zero_grad()
        cs.definition_sparsity_loss(lam=1., free_size=0).backward()
        optimizer.step()
    print(f'\ncertificate 13: after 30 steps {store.values.detach().tolist()}')
    assert bool((store.values.detach().abs() <= .02).all())
    order_zero = getattr(cs, 'order_zero', None)
    if order_zero is not None:
        penalty = cs.definition_sparsity_loss(lam=1., free_size=0)
        assert torch.autograd.grad(penalty, order_zero.value, allow_unused=True)[0] is None


def test_cert_6_the_symbols_leg_carries_each_symbols_two_lanes_and_no_code():
    """Certificate 6, its first clause over the leg: the symbolic leg is
    each symbol's two lanes, the product of its members' (here one member at
    (.7, .2)), and nothing else: no code, place or time of a concept, and no
    concept's identity. The gradient reaches the concept field."""
    from types import SimpleNamespace
    import Language
    cs, store, ids, rows = _small_space()
    word, cats, dogs, green, _ = rows
    cs.add_concept_edge(cats, word, 2.)
    inventory = torch.tensor([word, cats])
    field = _member_lanes(cs, [word, cats], word, (.7, .2)).requires_grad_()
    carrier = SimpleNamespace(_concept_activations=field, _concept_inventory_rows=inventory)
    ss = SimpleNamespace(conceptualSpace=cs,
                         _publish_symbol_snapshot=lambda event, **_kw: SimpleNamespace(event=event))
    leg = Language.SymbolSpace.forward_concept_to_symbol(ss, carrier)
    view = store.bind(inventory)
    print(f'\ncertificate 6: leg {tuple(leg.event.shape)}, lanes {leg.event[0].tolist()}, '
          f'symbols {leg._symbol_ids.tolist()}')
    assert leg.event.shape == (1, len(view._things), 2) == (1, 1, 2)
    torch.testing.assert_close(leg.event[0, 0], torch.tensor([.7, .2]))
    assert leg._symbol_ids.tolist() == [[view._things[0][1]]]
    assert set(vars(leg)) == {'event', '_symbol_ids'}
    leg.event[..., 0].sum().backward()
    assert float(field.grad[0, 0, 0, 0]) == 1.
    # No symbol read: no leg.
    carrier = SimpleNamespace(_concept_activations=field.detach(), _concept_inventory_rows=torch.tensor([dogs, green]))
    assert Language.SymbolSpace.forward_concept_to_symbol(ss, carrier) is None


# Certificate 18: the blank object and the meta concept (rule 20).

def _word_space():
    from test_cs_sparse_weights import _cs
    from nomination_fixtures import recurrent_word
    cs = _cs(nS=64, order=3, order_reservations=(0, 1, 1))
    cs.declare_word_classes()
    word = recurrent_word(cs.interpret, [7], [1], form='felix')
    return cs, word


def test_cert_18_a_word_with_no_object_gets_one_blank_object_and_one_meta_concept():
    from ConceptPlacement import positions
    cs, word = _word_space()
    alloc = cs._concept_allocator
    assert cs.interpret.no_object(word) and not cs.definitions.objects(word)
    before = set(alloc.concepts)
    obj, row, order = cs.interpret.reading(word)
    # One new concept with a row (the definition's relation takes a handle
    # of its own, with no row): the blank object, one order above its word,
    # with no members.
    made = {cid for cid in set(alloc.concepts) - before if cs._concept_row_of(cid) is not None}
    assert made == {obj} and 'blank' in alloc.symbol_types[obj]
    assert order == cs._concept_source_order(word) + 1 == 1
    assert row == cs._concept_row_of(obj) and cs._row_order(row) == 1
    assert not alloc.refs(obj, 'part') and not cs.concept_parts(obj)
    # The meta concept: the word's definition of the blank.
    assert cs.definitions.objects(word) == (obj,) and cs.definitions.row(word, obj) is not None
    assert not cs.interpret.no_object(word)
    # Its vector is zero, and a read that normalizes it gives zero, not NaN.
    vector = positions(cs)[row]
    assert float(vector.abs().sum()) == 0.
    assert float(cs.similarity_codebook.trust.content[row].abs().sum()) == 0.
    normalized = torch.nn.functional.normalize(vector, dim=-1)
    assert bool(torch.isfinite(normalized).all()) and float(normalized.abs().sum()) == 0.
    # Interpreting it again makes nothing new.
    concepts, definitions = set(alloc.concepts), len(alloc.parts)
    assert cs.interpret.reading(word) == (obj, row, order)
    assert cs.interpret.forward(word) == obj
    assert set(alloc.concepts) == concepts and len(alloc.parts) == definitions


def test_cert_18_every_object_found_or_made_for_a_word_is_a_member_of_objects():
    cs, word = _word_space()
    alloc = cs._concept_allocator
    objects = alloc.declared_classes['objects']
    kind_row = cs._concept_row_of(objects)
    assert cs._row_order(kind_row) == 2
    obj, row, _ = cs.interpret.reading(word)
    # Through the symbol that holds the object, as any kind's member.
    table = alloc.symbol_table()
    symbol = table.fit(obj)
    assert symbol is not None and table.members(symbol) == [obj]
    assert cs.reads_concept(kind_row, row)
    # An object found for a word (a definition to an existing object) too.
    from nomination_fixtures import recurrent_word
    other = recurrent_word(cs.interpret, [9], [1], form='ciao')
    found = cs.new_concept(order=1)
    cs.interpret.define(other, found)
    assert cs.reads_concept(kind_row, cs._concept_row_of(found))
    # No pending name is left in MTM.
    assert not any(isinstance(key, tuple) and key[:1] == ('name',) for key in cs.mtm.entries)
    assert not hasattr(cs.mtm, 'named') and not hasattr(cs.interpret, '_pending')


# Step 12: how MTM makes a symbol and a concept from a statement (rule 10).

def test_fit_a_symbol_holds_the_concept_and_a_maker_finds_it_first():
    cs, store, ids, rows = _small_space()
    table = cs._concept_allocator.symbol_table()
    felix, cats = ids[1], ids[2]                     # two concepts of order one
    first = cs.nominate_symbol(('candidate', 1), felix)
    # A second making for the same concept finds the symbol that holds it.
    assert cs.nominate_symbol(('candidate', 2), felix) == first and len(table) == 1
    assert table.fit(felix) == first and table.fit(felix, cats) is None


def test_same_thing_one_symbol_over_both_read_only_when_both_are_evidenced():
    cs, store, ids, rows = _small_space()
    table = cs._concept_allocator.symbol_table()
    felix, cat = ids[1], ids[2]
    alone = cs.nominate_symbol(('candidate', 1), felix)
    both = cs.same_thing(felix, cat)
    assert both != alone and sorted(table.members(both)) == sorted((felix, cat))
    # Its fit: stating it again, either way round, makes nothing new.
    assert cs.same_thing(cat, felix) == both and len(table) == 2
    # Concepts of two orders are never one symbol's constituents.
    assert cs.same_thing(ids[0], felix) is None
    # The plain product: both evidenced reads their product; one alone
    # reads nothing, on both lanes; the symbol of Felix alone is unchanged.
    lanes = torch.tensor([[.8, .1], [.5, .2]])
    read = table.read(lanes, [felix, cat], [alone, both])
    torch.testing.assert_close(read[1], torch.tensor([.4, .02]))
    torch.testing.assert_close(read[0], torch.tensor([.8, .1]))
    only = table.read(lanes[:1], [felix], [alone, both])
    assert float(only[1].abs().sum()) == 0. and torch.allclose(only[0], lanes[0])


def test_kind_entry_and_the_fit_of_a_concept():
    cs, store, ids, rows = _small_space()
    table = cs._concept_allocator.symbol_table()
    felix, ciao, kind, other = ids[1], ids[2], ids[3], ids[4]
    # "Felix is a cat", "Ciao is a cat": the kind's entries on its members'
    # symbols, found by their fit or made then.
    assert cs.kind_entry(kind, felix) == rows[3] and cs.kind_entry(kind, ciao) == rows[3]
    count = len(store._index)
    assert cs.kind_entry(kind, felix) == rows[3] and len(store._index) == count
    assert cs.reads_concept(rows[3], rows[1]) and cs.reads_concept(rows[3], rows[2])
    assert cs.kind_entry(felix, kind) is None        # not of a lower level
    # A concept fits when its entries read the same symbols: the kind over
    # Felix and Ciao fits them, and nothing fits Felix alone.
    assert cs.concept_fit([felix, ciao], 2) == kind and cs.concept_fit([ciao, felix], 2) == kind
    assert cs.concept_fit([felix], 2) is None and cs.concept_fit([felix, ciao], 1) is None
    cs.kind_entry(other, felix)
    assert cs.concept_fit([felix], 2) == other
    # A member with no symbol has no concept that reads it.
    assert cs.concept_fit([ids[0]], 1) is None or table.fit(ids[0]) is not None
