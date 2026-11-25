"""A form's links, found from the lists of its parts and characters, equal
the links found by looking through every stored form (item 1.75)."""
from test_item17_code import WORDS, coded


EXTRA = (b'a', b'ab', b'ba', b'bab', b'abc', b'cab', b'ar', b'arch', b'h', b'he', b'there is', b'cat cats')


def through_every_form(identity):
    words = list(identity.words)
    holders = {t: {w for w in words if w != t and t in w} for t in words}
    inside = {t: {w for w in words if w != t and w in t} for t in words}
    return holders, inside


def test_links_equal_the_look_through_every_stored_form():
    identity, *_ = coded(WORDS + EXTRA)
    holders, inside = through_every_form(identity)
    assert identity.holders == holders and identity.inside == inside
    assert any(identity.holders[w] for w in identity.words)


def test_links_after_a_load_equal_the_look_through_every_stored_form():
    identity, *_ = coded(WORDS)
    saved = identity.vocab_extras()
    other, *_ = coded(EXTRA[:3])
    other.load_vocab_extras(saved)
    for word in (b'arc', b'is this', b'c'):
        other.admit(word)
    holders, inside = through_every_form(other)
    assert other.holders == holders and other.inside == inside


def test_extent_read_joins_attested_parts_and_preserves_both_property_lanes(monkeypatch):
    import torch
    identity, basic, book, _ = coded((b'calaba', b'cabala', b'aba', b'ab', 'café'.encode()))
    data = list('calaba café cabala'.encode())
    raw = torch.tensor([data, list(reversed(data))])
    spans = torch.tensor([[(0, 6), (0, 0), (7, 12), (13, 19), (0, 3), (4, 6)],
                          [(0, 6), (1, 4), (0, 1), (4, 7), (2, 5), (0, 0)]])
    percepts = (None, None, book.primitive_properties, raw, None)
    def read():
        what, away = basic._occurrence_what(percepts, spans, (0, 1, 3))
        grad, = torch.autograd.grad(what.sum() + .3 * away.sum(), book.primitive_properties.members)
        return what.detach(), away.detach(), grad
    actual = read()
    with monkeypatch.context() as m:
        m.setattr(identity, 'occurrence_bank', lambda device: None)
        reference = read()
    for value, expected in zip(actual, reference):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert not torch.equal(actual[0][0, 0], actual[0][0, 3])


def test_device_part_join_preserves_tie_gradients_for_repeated_spelling():
    import torch
    from BasicPercepts import occurrence_parts
    raw = torch.tensor([[97, 98, 97, 98]])
    spans = torch.tensor([[(0, 4), (1, 3), (2, 4), (0, 0)]])
    spellings = (torch.tensor([[97], [98]]), torch.tensor([[97, 98]]))
    codes = torch.tensor([[1., 0.], [1., 1.], [1., 0.]], requires_grad=True)
    catalog = torch.eye(3)
    compiled = torch.compile(occurrence_parts, fullgraph=True, backend='eager')
    parts, tight = compiled(raw, spans, spellings, codes, catalog)
    torch.testing.assert_close(tight[0], torch.tensor([[1., 1., 1.], [1., 1., 0.],
                                                       [1., 1., 1.], [1., 1., 1.]]))
    reference = torch.stack([codes.amax(0), codes[:2].amax(0), codes.amax(0), codes[0] * 0])
    torch.testing.assert_close(parts[0], reference)
    torch.testing.assert_close(torch.autograd.grad(parts.sum(), codes)[0],
                               torch.autograd.grad(reference.sum(), codes)[0])


def test_packed_parts_keep_high_bits_and_pad_partial_limbs():
    import torch
    from types import SimpleNamespace
    from WordIdentity import WordIdentity
    layout = SimpleNamespace(total=129, limbs=3)
    values = torch.zeros(3, layout.total)
    values[0, [0, 63, 64, 127, 128]] = 1.
    values[1].fill_(1.)
    values[2, ::2] = .5
    expected = torch.tensor([[-(1 << 63) + 1, -(1 << 63) + 1, 1], [-1, -1, 1], [0, 0, 0]])
    packed = WordIdentity._pack(layout, values)
    assert torch.equal(packed, expected)
    assert torch.equal(WordIdentity._unpack(layout, packed), values.eq(1).long())
    assert WordIdentity._pack(layout, values[:0]).shape == (0, 3)


def test_batched_spelling_matches_scalar_inverse_with_both_code_widths():
    import torch
    identity, *_ = coded((b'calaba', b'cabala', b'ab', b'ba'))
    forms = torch.stack([identity.code(word) for word in identity.words])
    # Graded entries below one do not select the corresponding coordinates.
    forms = torch.cat((forms, forms[:1] * .75, torch.zeros_like(forms[:1])))
    for width in (identity.width, identity.total):
        inputs = forms[:, :width]
        assert identity.read_forms(inputs) == [identity.read(form) for form in inputs]
        assert identity.read_forms(inputs[:0]) == []


def test_ordinary_spelling_observations_preserve_the_order_and_counts_of_learning():
    serial, *_ = coded(())
    batched, *_ = coded(())
    forms = (b'cabala', b'aba', b'cabala', b'!', b'calaba', b'aba', b' ', b'caf\xc3\xa9', b'\xff')
    for raw in forms:
        serial.observe_nested(raw)
    batched.observe_forms(iter(forms))
    for name in ('words', 'word_rows', 'word_witnesses', 'nested', 'holders', 'inside'):
        assert getattr(serial, name) == getattr(batched, name), name
    for raw in forms:
        assert serial.form(raw).equal(batched.form(raw))


def test_input_ladder_finds_learned_parts_even_when_the_whole_has_no_word_row(monkeypatch):
    import torch
    from BasicPercepts import parts_for_units

    identity, *_ = coded((b'ell', b'll', b'he', b'!'))
    raw = torch.tensor([list(b'hell!'), list(b' hell')])
    spans = torch.tensor([[[0, 4], [1, 4], [2, 4], [0, 0]], [[1, 5], [2, 5], [3, 5], [0, 0]]])
    assert b'hell' not in identity.words
    expected = [[identity.atoms[atom] for atom in identity.atoms_of(word)]
                for word in (b'hell', b'ell', b'll', b'')]
    def word_lookup(*args):
        raise AssertionError('the input part read asked for a whole word')
    monkeypatch.setattr(identity, 'parts', word_lookup)
    monkeypatch.setattr(identity, 'atoms_of', word_lookup)
    spellings, rows, _, ranks = identity.occurrence_bank(raw.device)
    compiled = torch.compile(parts_for_units, fullgraph=True, backend='eager')
    for result in (identity.located_parts(raw, spans, -1), compiled(raw, spans, spellings, rows, ranks, -1)):
        addresses, counts = result
        for batch in range(2):
            for unit, parts in enumerate(expected):
                assert addresses[batch, unit, :len(parts)].tolist() == parts
                assert counts[batch, unit] == len(parts)
                assert addresses[batch, unit, len(parts):].eq(-1).all()


def test_disjoint_reading_preserves_repeats_and_shifted_byte_intervals():
    import torch
    from BasicPercepts import disjoint_parts

    identity, *_ = coded((b'sho', b'pping', b'shopping', b'aba', 'é'.encode()))
    text = 'shopping ababa éé'.encode()
    raw = torch.tensor([list(text + b' '), list(b' ' + text)])
    spans = torch.tensor([[[0, 8], [9, 14], [15, 19], [0, 0]],
                          [[1, 9], [10, 15], [16, 20], [0, 0]]])
    groups, rows, _, ranks = identity.occurrence_bank(raw.device)
    compiled = torch.compile(disjoint_parts, fullgraph=True, backend='eager')
    from BasicPercepts import pack_disjoint_units
    flat, counts, locations = identity.sentence_cover(raw, spans, -1)
    assert flat.shape == raw.shape and locations.shape == (*raw.shape, 2)
    packed, extents = pack_disjoint_units(flat, counts, locations, -1, raw.shape[1])
    eager = packed, counts, extents
    for actual, reference in zip(compiled(raw, spans, groups, rows, ranks, -1), eager):
        torch.testing.assert_close(actual, reference)
    parts, counts, extents = eager
    for b in range(len(raw)):
        for w, expected in enumerate(((b'sho', b'pping'), (b'aba', b'b', b'a'), ('é'.encode(),) * 2)):
            start, end = spans[b, w].tolist()
            n = int(counts[b, w])
            ranges = extents[b, w, :n].tolist()
            assert [bytes(raw[b, lo:hi].tolist()) for lo, hi in ranges] == list(expected)
            assert ranges[0][0] == start and ranges[-1][1] == end
            assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
            assert [identity._fixed_bytes[row] for row in parts[b, w, :n].tolist()] == list(expected)
        assert counts[b, 3] == 0 and parts[b, 3].eq(-1).all()


def test_inverse_follows_ordered_native_children_after_restore(monkeypatch):
    from FieldReverse import spelling

    words = (b'a', b'b', b'ab', b'abab', b'aabb')
    identity, basic, _, rows = coded(words)
    saved = identity.store.vocab_extras()
    other, restored, *_ = coded(words)
    other.store.load_vocab_extras(saved)
    for current, reader in ((identity, basic), (other, restored)):
        def forbidden_name(row):
            raise AssertionError('inverse requested the whole spelling')
        monkeypatch.setattr(reader, 'form_of', forbidden_name)
        assert spelling(reader, rows[b'abab']) == b'abab'
        assert spelling(reader, rows[b'aabb']) == b'aabb'
        # A deliberately reversed native definition changes the inverse even
        # though the named form and its orthographic lookup are unchanged.
        root = current.word_rows[b'aabb']
        current.store.part_groups[root] = tuple(reversed(current.store.part_groups[root]))
        assert spelling(reader, rows[b'aabb']) != b'aabb'


def test_compiled_concatenation_keeps_null_bytes_and_empty_forms():
    import torch
    from FieldReverse import concatenate_parts

    parts = torch.tensor([[[97, 0], [98, -1], [98, -1]], [[-1, -1]] * 3])
    expected = torch.tensor([[97, 0, 98, 98, -1], [-1] * 5])
    result = torch.compile(concatenate_parts, fullgraph=True, backend='eager')(parts, 5)
    torch.testing.assert_close(result, expected)


def test_disjoint_cover_decodes_utf8_within_each_cut_when_escaped_bytes_are_known():
    import torch
    from BasicPercepts import disjoint_occurrences, pack_disjoint_units
    from WordIdentity import text_of

    raw = b'x' + 'é中😀'.encode() + b'\xff\xc0\xed\xa0\x80\xf4\x90\x80\x80'
    identity, *_ = coded((b'x', b'x\xc3', 'é'.encode(), '中'.encode(), '😀'.encode()))
    for byte in raw:
        identity._atom(('part', text_of(bytes([byte]))))
    # Compare every possible interval to the lesson's decoder. In particular
    # the learned part x + escaped C3 must stop matching when C3 begins é.
    intervals = [(a, z) for a in range(len(raw)) for z in range(a + 1, len(raw) + 1)]
    surface = torch.tensor(list(raw)).expand(len(intervals), -1)
    spans = torch.tensor(intervals)[:, None]
    flat, counts, extents = identity.sentence_cover(surface, spans, -1)
    keys, rows, _, ranks = identity.occurrence_bank(surface.device, escaped_elements=True)
    compiled = torch.compile(disjoint_occurrences, backend='eager', fullgraph=True)
    for value, expected in zip(compiled(surface, spans, keys, rows, ranks, -1, escaped_elements=True),
                               (flat, counts, extents), strict=True):
        torch.testing.assert_close(value, expected)
    from BasicPercepts import escaped_part_starts
    first = torch.compile(escaped_part_starts, backend='eager', fullgraph=True)(surface, spans, keys)
    evidence, distinct = identity.located_parts(surface, spans, -1)
    parts, spans = pack_disjoint_units(flat, counts, extents, -1, len(raw))
    for b, (lo, hi) in enumerate(intervals):
        available = [identity.atoms[atom] for atom in identity.atoms_of(raw[lo:hi]) if atom in identity.atoms]
        assert evidence[b, 0, :distinct[b, 0]].tolist() == available
        assert set(rows[first[b, 0] < len(raw)].tolist()) == set(available)
        cover = identity.ordered_cover(raw[lo:hi])
        n = counts[b, 0]
        assert parts[b, 0, :n].tolist() == [identity.atoms[part] for part, _, _ in cover]
        assert spans[b, 0, :n].tolist() == [[lo + start, lo + end] for _, start, end in cover]


def test_new_element_seats_match_rereading_every_utf8_cut():
    import torch
    from BasicPercepts import fill_new_elements
    from WordIdentity import text_of
    identity, *_ = coded((b'ab',))
    raw = 'abé中🙂z'.encode() + bytes((0xff, 0xc0, 0xaf))
    cuts = [(lo, hi) for lo in range(len(raw)) for hi in range(lo + 1, len(raw) + 1)]
    surface = torch.tensor(list(raw)).expand(len(cuts), -1)
    spans = torch.tensor(cuts)[:, None]
    flat, counts, extents, forms = identity.sentence_cover(surface, spans, -1, identify=True, unseen_elements=True)
    missing = sorted({char for lo, hi in cuts for char in text_of(raw[lo:hi])
                      if ('part', char) not in identity.atoms})
    table = flat.new_tensor([(ord(char), identity._atom(('part', char))) for char in missing]).T.contiguous()
    fill = torch.compile(fill_new_elements, backend='eager', fullgraph=True)
    installed = fill(surface.to(torch.uint8), flat, extents, *table)
    reread = identity.sentence_cover(surface, spans, -1, identify=True)
    for value, expected in zip((installed, counts, extents, forms), reread, strict=True):
        torch.testing.assert_close(value, expected, rtol=0, atol=0)


def test_full_unit_matches_count_exposures_without_becoming_spelling_operands():
    import torch
    from BasicPercepts import disjoint_occurrences

    identity, *_ = coded((b'sho', b'pping', b'shopping', b'x'))
    identity.admit_part(b'op')
    raw = torch.tensor([list(b'shopping x shopping op y')])
    spans = torch.tensor([[(0, 8), (9, 10), (11, 19), (20, 22), (23, 24), (0, 0)]])
    ids, counts, extents, forms = identity.sentence_cover(raw, spans, -1, identify=True)
    keys, rows, _, ranks, native = identity._occurrences(raw.device, False)
    compiled = torch.compile(disjoint_occurrences, backend='eager', fullgraph=True)
    for value, reference in zip(compiled(raw, spans, keys, rows, ranks, -1, form_rows=native),
                                (ids, counts, extents, forms), strict=True):
        torch.testing.assert_close(value, reference)
    shopping, x = (identity.word_rows[name] for name in (b'shopping', b'x'))
    assert forms.tolist() == [[shopping, x, shopping, -1, -1, -1]]
    assert ids[0, :2].tolist() == [identity.atoms[('part', name)] for name in ('sho', 'pping')]
    before = identity._witness_counts.clone()
    identity.observe_rows(forms)
    before[shopping] += 2
    before[x] += 1
    torch.testing.assert_close(identity._witness_counts, before, atol=0, rtol=0)


def test_new_reading_part_is_local_until_the_next_packed_block(monkeypatch, eager_reading):
    from BasicPercepts import BasicPercepts
    from Models import BaseModel
    from What import What
    from data import Data

    seen = []
    original = BasicPercepts._occurrence_what
    def capture(self, percepts, spans, wholes, **options):
        result = original(self, percepts, spans, wholes, **options)
        if not seen[-1]:
            seen[-1].update(spans=spans.detach().clone(), lanes=result[0].detach().clone())
        return result
    monkeypatch.setattr(BasicPercepts, '_occurrence_what', capture)
    model, _ = BaseModel.from_config('data/XOR_grammar.xml', data=Data())
    data = model.inputSpace.data
    optimizer = model.getOptimizer()
    try:
        for _ in range(2):
            seen.append({})
            with data.runtime_batch(['abc', 'ab']):
                data.has_supervised_outputs = False
                source = model.inputSpace.prepInput(['abc', 'ab'])
                model.runBatch(train=True, optimizer=optimizer, batchSize=2, split='runtime',
                               questions=tuple(What.present(i, split='runtime') for i in range(2)),
                               batch_override=(source, source.new_empty(2, 0)))
            model.End()
            model.dispatch_per_row_reset([True, True])
        first, next_block = seen
        assert [0, 2] not in first['spans'][0].tolist()
        assert [0, 2] in first['spans'][1].tolist()
        assert [0, 2] in next_block['spans'][0].tolist()
        left = first['spans'][0].tolist().index([0, 3])
        right = next_block['spans'][0].tolist().index([0, 3])
        width = model.perceptualSpace.percept_store.identity.width
        assert (first['lanes'][0, left, :width] < next_block['lanes'][0, right, :width]).any()
    finally:
        model.End()


def test_later_packed_sentence_cannot_teach_an_earlier_opening(monkeypatch, eager_reading):
    from BasicPercepts import BasicPercepts
    from Models import BaseModel
    from What import What
    from data import Data

    captured = []
    original = BasicPercepts._occurrence_what
    def capture(self, percepts, spans, wholes, **options):
        result = original(self, percepts, spans, wholes, **options)
        if not captured:
            captured.append((spans[0].tolist(), result[0][0].detach().clone()))
        return result
    monkeypatch.setattr(BasicPercepts, '_occurrence_what', capture)
    model, _ = BaseModel.from_config('data/MM_corpus_lessons.xml', data=Data())
    data = model.inputSpace.data
    try:
        with data.runtime_batch(['abc ab abc']):
            data.has_supervised_outputs = False
            source = model.inputSpace.prepPackedInput([['abc', 'ab', 'abc']])
            model.runBatch(train=True, optimizer=model.getOptimizer(), batchSize=1, split='runtime',
                           questions=(What.present(0, split='runtime'),),
                           batch_override=(source, source.new_empty(1, 0)))
        spans, codes = captured[0]
        assert [0, 2] not in spans
        assert [7, 9] in spans
        width = model.perceptualSpace.percept_store.identity.width
        first, last = (codes[spans.index(interval), :width] for interval in ([0, 3], [7, 10]))
        assert (first <= last).all() and (first < last).any()
    finally:
        model.End()


def test_promoted_pi_changes_at_the_observing_streams_opening(monkeypatch, eager_reading):
    import torch
    from BasicPercepts import BasicPercepts, held_at
    from Models import BaseModel
    from What import What
    from data import Data

    model, _ = BaseModel.from_config('data/XOR_grammar.xml', data=Data())
    data = model.inputSpace.data
    optimizer = model.getOptimizer()
    basic = model._concept_owner().basic_percepts

    def present(pages):
        with data.runtime_batch(pages):
            data.has_supervised_outputs = False
            source = model.inputSpace.prepInput(pages)
            model.runBatch(train=True, optimizer=optimizer, batchSize=len(pages), split='runtime',
                           questions=tuple(What.present(i, split='runtime') for i in range(len(pages))),
                           batch_override=(source, source.new_empty(len(pages), 0)))
        model.End()
        model.dispatch_per_row_reset([True] * len(pages))

    try:
        for _ in range(4):
            present(['abc', 'abc'])
            promoted = [r for r, name in basic.forms.items() if name == b'abc' and basic.promoted[r]]
            if promoted:
                break
        row, = promoted
        before = basic.bank_codes((row,))[0].clone()
        original = BasicPercepts.lookup
        reads = []

        def capture(self, percepts, brackets, rows, **options):
            bank = original(self, percepts, brackets, rows, **options)
            if self is basic and bank.revisions is not None and row in bank.rows and not reads:
                at = bank.rows.index(row)
                revised = bank.revisions
                version = revised.at(bank.span[..., 0])[revised.indices.tolist().index(at)]
                members = held_at(bank, bank.table)[0][at]
                for b in range(2):
                    index = bank.span[b].tolist().index([0, 3])
                    reads.append((version[b, index].clone(), members[b, index].item()))
            return bank
        monkeypatch.setattr(BasicPercepts, 'lookup', capture)
        present(['abc', 'abc ab'])
        after = basic.bank_codes((row,))[0]
        assert not torch.equal(before, after)
        torch.testing.assert_close(reads[0][0], before, atol=0, rtol=0)
        torch.testing.assert_close(reads[1][0], after, atol=0, rtol=0)
        assert [score for _, score in reads] == [1., 1.]
    finally:
        model.End()


def test_page_scoped_pi_and_inverse_match_independent_gradients():
    import torch
    from types import SimpleNamespace
    from BasicPercepts import held_at
    from BlockPercepts import RowRevisions
    from FieldReverse import spelling_scores

    # The two pages change different rows at different byte offsets. Binary
    # coordinates deliberately exercise ties in the least-of-greater fold.
    rng = torch.Generator().manual_seed(112)
    coords = torch.randint(2, (4, 2, 12), generator=rng).float()
    versions = torch.randint(2, (2, 2, 2, 2, 12), generator=rng).float()
    revised = RowRevisions((0, 3), torch.tensor([0, 3]), torch.tensor([[0, 3], [0, 5]]), versions)
    offsets = torch.tensor([[0, 3, 5], [0, 3, 5]])
    spans = torch.stack((offsets, offsets + 1), -1)
    b = SimpleNamespace(codes=coords, identity_width=6, width=12, span=spans, revisions=revised)
    data = torch.randint(2, (2, 3, 12, 2), generator=rng).float().requires_grad_()
    held = torch.rand(4, 2, 3, generator=rng, requires_grad=True)
    selected = [3, 1, 0]
    actual = (*held_at(b, data), spelling_scores(held, coords[:, 0], selected, offsets, revised))
    expect = [torch.empty_like(x) for x in actual]
    for stream, starts in enumerate(([0, 3], [0, 5])):
        for i, position in enumerate((0, 3, 5)):
            when = int(position >= starts[1])
            codes = coords.clone()
            codes[[0, 3]] = versions[:, stream, when]
            cell = data[stream:stream+1, i:i+1]
            for j, x in enumerate(held_at(b, cell, codes=codes)):
                expect[j][:, stream, i] = x[:, 0, 0]
            expect[-1][stream, i] = spelling_scores(held[:, stream:stream+1, i:i+1],
                                                    codes[:, 0], selected, offsets[stream:stream+1, i:i+1])[0, 0]
    for a, e in zip(actual, expect):
        torch.testing.assert_close(a, e)
    def gradients(outputs):
        return torch.autograd.grad(sum(x.sum() for x in outputs), (data, held))
    for a, e in zip(gradients(actual), gradients(expect)):
        torch.testing.assert_close(a, e)


def test_repaired_atom_coordinates_stay_with_their_witnesses_until_consolidation():
    import torch
    from BasicPercepts import occurrence_parts
    from BlockPercepts import revisions

    identity, basic, _, rows = coded([b'aab', b'baa', b'bab'])
    row = rows[b'bab']
    before = basic.bank_codes([row])[0]
    surface = torch.tensor([list(b'bab bab')] * 2)
    spans = torch.tensor([[[0, 3], [4, 7]]] * 2)

    def read(block=None):
        spellings, ids, catalog, _ = identity.occurrence_bank(surface.device)
        basis = basic.native._basis.lookup_rows(ids)[..., :identity.width]
        return torch.cat(occurrence_parts(surface, spans, spellings, basis, catalog,
                             available_from=None if block is None else block.openings(ids),
                             block=block, addresses=ids), -1)

    original = read()
    prior = len(identity.added)
    repairs = []
    identity.observe_forms([b'baab'], repairs=repairs)
    after = basic.bank_codes([row])[0]
    assert not torch.equal(before, after)
    block = identity.block_part_visibility(prior, {'baab': {1: 4}}, 2, surface.device,
                                           sentence_starts=[[0, 4], [0, 4]], repairs=repairs)
    versions = revisions(basic, [row], block).at(spans[..., 0])[0]
    torch.testing.assert_close(versions[0], before.expand_as(versions[0]), atol=0, rtol=0)
    torch.testing.assert_close(versions[1, 0], before, atol=0, rtol=0)
    torch.testing.assert_close(versions[1, 1], after, atol=0, rtol=0)
    scoped, consolidated = read(block), read()
    assert not torch.equal(consolidated, original)
    torch.testing.assert_close(scoped[0], original[0], atol=0, rtol=0)
    torch.testing.assert_close(scoped[1, 0], original[1, 0], atol=0, rtol=0)
    torch.testing.assert_close(scoped[1, 1], consolidated[1, 1], atol=0, rtol=0)


def test_preexisting_part_becomes_a_holder_only_on_the_observing_page():
    import torch
    from BlockPercepts import revisions

    identity, basic, _, rows = coded([b'b'])
    identity.admit_part(b'abc')
    before = basic.bank_codes([rows[b'b']])[0]
    prior_parts, prior_forms = len(identity.added), len(identity.words)
    identity.admit(b'abc')
    assert len(identity.added) == prior_parts
    after = basic.bank_codes([rows[b'b']])[0]
    assert (after[1] > before[1]).any()
    block = identity.block_part_visibility(prior_parts, {'abc': {1: 0}}, 2, before.device,
                                           prior_forms=prior_forms)
    versions = revisions(basic, [rows[b'b']], block).at(torch.zeros(2, 1, dtype=torch.long))[0, :, 0]
    torch.testing.assert_close(versions[0], before, rtol=0, atol=0)
    torch.testing.assert_close(versions[1], after, rtol=0, atol=0)


def test_containment_accidents_preserve_direction_and_linked_parts_in_both_limbs():
    import torch
    from WordIdentity import containment_accidents

    # One coordinate is bit 63, another is bit 64. Rows 1 and 2
    # have identical codes but no attested link to one another.
    bit = -(1 << 63)
    codes = torch.tensor([[bit, 0], [bit, 1], [bit, 1], [0, 1]])
    rows = torch.tensor([2, 1])
    # Only form 1 really contains form 0.
    linked = torch.tensor([4])
    matches = containment_accidents(codes, rows, linked)
    assert matches.tolist() == [[[True, True, False, True], [False, False, True, True]],
                                [[False, True, False, False], [False, False, True, False]]]


def test_tensor_repair_selects_private_coordinates_and_reports_exhausted_holders():
    from WordIdentity import distinguishing_coordinates
    import torch

    # Both pages must distinguish the first two outers. Their local usage
    # differs, and the final outer on page one has no available coordinate.
    lacking = torch.tensor([[[1, 1, 0], [0, 0, 1], [0, 0, 0]]] * 2, dtype=torch.bool)
    kept = torch.ones(2, 3, dtype=torch.bool)
    opposed = torch.zeros_like(kept)
    usage = torch.tensor([[0, 1, 0], [1, 0, 0]])
    pending = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    given, remaining = distinguishing_coordinates(lacking, kept, opposed, torch.zeros_like(usage), usage, pending)
    assert given.tolist() == [[True, False, True], [False, True, True]]
    assert remaining.tolist() == [[False, False, False], [False, False, True]]


def test_simultaneous_collision_repairs_match_separate_pages_even_outside_the_consolidated_code():
    import torch
    from BlockPercepts import revisions

    prior = [b'aab', b'baa', b'bab']
    pages = [(b'baba',), (b'baab',)]
    isolated = []
    for page in pages:
        identity, basic, _, rows = coded(prior)
        identity.observe_forms(page)
        isolated.append(basic.bank_codes([rows[b'bab']])[0])
    identity, basic, _, rows = coded(prior)
    prior_parts, prior_forms = len(identity.added), len(identity.words)
    edits = []
    identity.observe_forms([w for page in pages for w in page], repairs=edits)
    seen = {'baba': {0: 0}, 'baab': {1: 0}}
    block = identity.block_part_visibility(prior_parts, seen, 2, basic.value_codes.device,
                                           repairs=edits, prior_forms=prior_forms)
    versions = revisions(basic, [rows[b'bab']], block).at(torch.zeros(2, 1, dtype=torch.long))[0, :, 0]
    torch.testing.assert_close(versions, torch.stack(isolated), atol=0, rtol=0)
    assert (isolated[1][0] > basic.bank_codes([rows[b'bab']])[0, 0]).any()
