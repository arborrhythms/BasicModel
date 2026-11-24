"""Item 1.7, rule 76: the table of attended percepts (sigma-pi §21,
certificates 61 to 63, and 33 (b)).

Certificate 61 is read on the lesson's sixteen sentences: each sentence's
occurrences (each part's event, each whole's span, each reading's bracket)
with their codes, the what columns of the bank's table, the rows of the
first Pi read from those codes by ``BasicPercepts.read_points``, and a
reading's value the greatest over the occurrences inside its bracket;
compared with ``read_bank`` on the same sentence, which reads each row's
members at each attended percept. The comparison is printed; it asserts
only that it was made.
"""
import torch


def compare_table(basic, percepts, bank):
    """Each row of the bank read at each reading from the what columns of
    the bank's table (each occurrence's code on its for lane) by
    ``read_points``, beside ``read_bank``'s; returns the counts and every
    difference ``(row, form, reading text, read_bank lanes, table lanes,
    kind)``."""
    native = basic.native
    leaves = basic._leaves()
    ids, spans = percepts[0][0].tolist(), percepts[1][0].tolist()
    text = {}
    for part, (start, end) in zip(ids, spans):
        if part < 0 or end <= start:
            continue
        cursor = start
        for leaf in leaves(part):
            for byte in native.bytes_for(leaf):
                text[cursor] = byte
                cursor += 1
    raw_of = lambda a, z: bytes(text.get(i, 32) for i in range(a, z))
    with torch.no_grad():
        read = basic.read_bank(bank).values[:, 0]                          # [R, K, 2]
    brackets = bank.where[0].tolist()
    real = bank.span[0, :, 1] > bank.span[0, :, 0]
    occurrences = [tuple(span) for span in bank.span[0][real].tolist()]
    points = bank.table[0][real][:, :bank.width, 0]                       # [O, width]
    rows = list(bank.rows)
    plain = [r for r in rows if r not in basic.place_rows and r not in basic.conjunctions]
    table = read.new_zeros(len(rows), len(brackets), 2)
    if plain:
        boundary = basic.read_points(points[None].to(basic.value_codes), plain)  # [2, R, 1, O, 2]
        part, whole = boundary[0, :, 0, :, 0], boundary[1, :, 0, :, 0]   # [R, O]
        against = torch.stack([basic.against(r) for r in plain]).detach()[:, :basic.identity_width] > .5
        hit = (points[None, :, :basic.identity_width] > .5) & against[:, None]
        added = hit.any(-1).to(part)
        value = torch.stack((part * whole, part * torch.maximum(1 - whole, added)), -1)   # [R, O, 2]
        for k, (a, z) in enumerate(brackets):
            inside = torch.tensor([a <= s and e <= z for s, e in occurrences])
            if bool(inside.any()):
                at = value[:, inside].amax(1)
                for n, r in enumerate(plain):
                    table[rows.index(r), k] = at[n]
    differences, equal = [], 0
    for n, r in enumerate(rows):
        form = basic.forms.get(r)
        for k, (a, z) in enumerate(brackets):
            if z <= a:
                continue
            old, new = read[n, k], table[n, k]
            if torch.allclose(old, new, atol=1e-5):
                equal += 1
                continue
            kind = ('place' if r in basic.place_rows else 'conjunction' if r in basic.conjunctions else
                    'for' if abs(float(old[0] - new[0])) > 1e-5 else 'against')
            differences.append((r, form, raw_of(a, z), [round(float(v), 4) for v in old],
                                [round(float(v), 4) for v in new], kind))
    return dict(equal=equal, differences=differences, occurrences=len(occurrences))


def test_61_the_table_against_the_present_read(monkeypatch):
    """Certificate 61, recorded: the rows of the first Pi read from the
    occurrences' codes against ``read_bank``, at each reading of each of the
    lesson's sixteen sentences."""
    from test_item17_order_zero import _read_lesson
    from BasicPercepts import BasicPercepts
    seen, results = set(), []
    original = BasicPercepts.lookup

    def recorded(self, percepts, brackets, rows, **options):
        bank = original(self, percepts, brackets, rows, **options)
        key = (tuple(percepts[0][0].tolist()), tuple(map(tuple, percepts[1][0].tolist())))
        if key not in seen and len(bank.rows):
            seen.add(key)
            results.append(compare_table(self, percepts, bank))
        return bank
    monkeypatch.setattr(BasicPercepts, 'lookup', recorded)
    model, patch = _read_lesson(16)
    try:
        pass
    finally:
        model.End()
        patch.undo()
    equal = sum(r['equal'] for r in results)
    differences = [d for r in results for d in r['differences']]
    kinds = {}
    for d in differences:
        kinds[d[-1]] = kinds.get(d[-1], 0) + 1
    print(f'\ncertificate 61: {len(results)} sentences, occurrences {[r["occurrences"] for r in results]}, '
          f'{equal} (row, reading) equal, {len(differences)} differ, by kind {kinds}')
    for d in differences:
        print('  row', d[0], d[1], 'at', repr(d[2]), 'read_bank', d[3], 'table', d[4], d[5])
    assert results


def _word_percept(model, word):
    """The basic percept row a word's concept was brought by."""
    from test_item17_looks import word_rows
    from Spaces import _concept_alloc_of
    owner = model._concept_owner()
    concept, = word_rows(model, word)
    return next(p for p, r in _concept_alloc_of(owner).brought.items() if r == concept), concept


def test_62_33b_what_is_where(tmp_path, monkeypatch):
    """Certificate 62. (b) A sentence that holds one word twice gives two
    attended percepts with one what and two wheres. (a) A row that keeps an
    entry on the where wires of the first reads its what there and less at
    the second; with the entry at nothing it reads both alike. (c) A request
    for its concept, carried down, lands most on the attended percept that
    has both its what and its where; and (33 (b)) a gradient from the row's
    value there reaches the row of ``WhereRegistry.code`` at that place."""
    import Salience
    from BasicPercepts import attended, WHERE_WIRES
    from test_item17_looks import nominated_lesson, held_banks
    from test_item18_expect import read
    model = nominated_lesson(tmp_path, monkeypatch)
    try:
        banks = held_banks(monkeypatch)
        read(model, 'alpha gamma alpha')
        owner, bank = model._concept_owner(), banks[-1]
        basic = owner.basic_percepts
        row, concept = _word_percept(model, 'alpha')
        r = bank.rows.index(row)
        with torch.no_grad():
            perception = basic.read_bank(bank)
        # (b) The occurrences at which the row reads fully: one what, two wheres.
        at = perception.events[r, 0, :, 0]
        full = (at >= at.max() - 1e-6) & (at > 0)
        index = perception.index[0][full]
        spans = bank.span[0, index]
        words = [i for i, (a, z) in zip(index.tolist(), spans.tolist()) if z - a == 5]
        first, second = sorted(words, key=lambda i: int(bank.span[0, i, 0]))[:2]
        what = bank.table[0, :, :bank.width]
        where = bank.table[0, :, bank.width:bank.width + WHERE_WIRES, 0]
        print(dict(full=spans.tolist(), first=bank.span[0, first].tolist(), second=bank.span[0, second].tolist(),
                   where_first=where[first].tolist(), where_second=where[second].tolist()))
        assert torch.equal(what[first], what[second])
        assert not torch.equal(where[first], where[second])
        # (a) Entries on the where wires of the first: each wire's two lanes.
        c = where[first].clamp(1e-6, 1 - 1e-6)
        before = perception.events[r, 0]
        slot = {int(o): a for a, o in enumerate(perception.index[0].tolist())}
        assert torch.allclose(before[slot[first]], before[slot[second]])
        with torch.no_grad():
            basic.where_entries[row] = torch.cat((c, 1 - c))
        try:
            look = attended(bank)
            perception = basic.read_bank(bank, look=look)
            kept = perception.events[r, 0]
            print(dict(at_first=kept[slot[first]].tolist(), at_second=kept[slot[second]].tolist(),
                       before=before[slot[first]].tolist()))
            # One at its own place, as with the entry at nothing; less at the other.
            assert torch.allclose(kept[slot[first]], before[slot[first]], atol=1e-5)
            assert float(kept[slot[first], 0]) > float(kept[slot[second], 0])
            # (33 (b)) The gradient from the row's value at an attended
            # percept reaches the where wires' row at its place, at the first
            # less than at the second: where a wire's lane equals the row's
            # entry the derivative passes on the lane at the tie alone.
            code = bank.registry.code
            grads = {}
            for name, at in (('first', first), ('second', second)):
                code.grad = None
                perception = basic.read_bank(bank, look=attended(bank))
                perception.events[r, 0, slot[at], 0].backward()
                address = bank.registry.slices['text'][0] + int(bank.span[0, at, 0])
                grad = code.grad.abs().sum(-1)
                grads[name] = (float(grad[address]), float(grad.sum() - grad[address]))
            print(dict(gradient_at_place_and_elsewhere=grads))
            assert grads['second'][0] > 0 and grads['second'][1] == 0
            assert grads['first'][0] < grads['second'][0]
            # (c) The request for its concept lands most on the first.
            request = torch.zeros(int(owner.nVectors), 1, 2)
            request[concept, 0, 0] = 1.
            # Ported (eighth pass): the share on the where wires (``band``)
            # was asserted greater at the first too; there is no band since
            # rule 79, the row's read holding its .what and .where together.
            with torch.no_grad():
                salient = Salience.salience(owner, bank, request)
            share = salient.events[0, :, 0] + salient.events[0, :, 1]
            print(dict(share_first=float(share[first]), share_second=float(share[second])))
            assert float(share[first]) > float(share[second])
        finally:
            with torch.no_grad():
                basic.where_entries[row] = 0.
    finally:
        model.End()


def test_63_what_a_look_reads_is_its_table(tmp_path, monkeypatch):
    """Certificate 63: the bank's tensors have one axis of occurrences and no
    axis of readings beside it; what a look reads of the percepts is
    ``[.., attended, ..]``, at most ``attendedPercepts`` of them, also where
    a sentence holds more occurrences than that."""
    import BasicPercepts
    from test_item17_looks import nominated_lesson, held_banks
    from test_item18_expect import read
    model = nominated_lesson(tmp_path, monkeypatch)
    try:
        banks = held_banks(monkeypatch)
        read(model, 'alpha gamma beta delta')
        owner, bank = model._concept_owner(), banks[-1]
        basic = owner.basic_percepts
        O = bank.token.shape[1]
        assert not hasattr(bank, 'native') and not hasattr(bank, 'band')
        # No member is read at the occurrences: the bank holds the rows'
        # codes, one row each, and the occurrences' table.
        assert not hasattr(bank, 'members')
        assert bank.codes.shape == (len(bank.rows), 2, bank.width)
        assert bank.table.shape[1] == O and bank.table.ndim == 4
        print(dict(occurrences=O, codes=tuple(bank.codes.shape), table=tuple(bank.table.shape)))
        capacity = O // 2
        monkeypatch.setattr(BasicPercepts, 'attended_capacity', lambda: capacity)
        with torch.no_grad():
            perception = basic.read_bank(bank)
        assert perception.events.shape[2] == capacity and perception.events.ndim == 4
        chosen = perception.index[0][perception.reading[0] >= 0]
        extents = (bank.span[0, :, 1] - bank.span[0, :, 0])
        print(dict(capacity=capacity, chosen_extents=sorted(extents[chosen].tolist(), reverse=True)))
        # Ported (eighth pass): the largest occurrences inside the sentence
        # came first. Since rule 79 the table is the most salient, a draw
        # among equals, with no order by extent; with nothing asked every
        # occurrence is equal, and the table is full.
        assert len(set(perception.index[0].tolist())) == capacity >= len(chosen)
    finally:
        model.End()


def test_learned_subword_extents_reach_the_open_bank_alongside_the_word(tmp_path, monkeypatch):
    from test_item17_looks import held_banks
    from test_item18_expect import lesson, read
    model = lesson(tmp_path, monkeypatch)
    try:
        banks = held_banks(monkeypatch)
        ws = model.wholeSpace
        with torch.no_grad():
            ws.begins_weight.fill_(-8.)
            ws.ends_weight.fill_(-8.)
            ws.atom_level.fill_(-8.)
            ws.begins_weight[ord('l')] = 8.
        for text, offset in (('hello', 0), (' hello', 1)):
            read(model, text)
            bank = banks[-1]
            intervals = [tuple(pair) for pair in bank.span[0].tolist()]
            for start, stop in ((0, 2), (2, 5), (0, 5)):
                span = (offset + start, offset + stop)
                assert intervals.count(span) == 1
                seat = intervals.index(span)
                assert bank.activity[0, seat].eq(1).all()
                assert bank.where[0, bank.token[0, seat]].tolist() == [offset, offset + 5]
    finally:
        model.End()
