"""Item 1.7, rule 76: the table of attended percepts (sigma-pi §21,
certificates 61 to 63).

Certificate 61 is read on the lesson's sixteen sentences: each sentence's
occurrences (each part's event, each word's bracket, each whole), each
with its code on two lanes, the rows of the first Pi read from those codes
by ``BasicPercepts.read_points``, and the reading's value the greatest over
the occurrences inside its bracket; compared with ``read_bank`` on the same
sentence. The comparison is printed; it asserts only that it was made.
"""
import torch


def _occurrence_code(basic, identity, raw, wholes):
    """An occurrence's code on its for lane: the parts half the greatest of
    its parts' codes, the wholes half the least of the wholes present over
    its span and its tightest whole (as ``subspace_bounds`` for a row)."""
    if identity is None:
        return None
    atoms = identity.words.get(raw) or identity.atoms_of(raw)
    rows = [identity.atoms[a] for a in atoms if a in identity.atoms]
    if not rows:
        return None
    parts = basic.part_codes(rows).amax(0)
    held = basic.codes.new_ones(basic.property_width)
    if wholes:
        held = basic.whole_codes(list(wholes)).amin(0)
    tightest = identity.tightest(rows)
    if tightest is not None:
        held = torch.minimum(held, tightest.to(held))
    return torch.cat((parts, held))


def compare_table(basic, percepts, bank):
    """Each row of the bank read at each reading from the codes of the
    sentence's occurrences, beside ``read_bank``'s; returns the counts and
    every difference ``(row, form, reading text, read_bank lanes, table
    lanes, kind)``."""
    native = basic.native
    identity = getattr(native, 'identity', None)
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
    events = bank.native.spans[0].tolist()
    brackets = bank.native.where[0].tolist()
    occurrences = [tuple(span) for span in events if span[1] > span[0]]
    occurrences += [tuple(b) for b in brackets if b[1] > b[0] and tuple(b) not in occurrences]
    keys = list(bank.native.keys)
    ws = [(i, key[1]) for i, key in enumerate(keys) if key[0] == 'ws']
    lit = bank.native.events[:, 0, ..., 0].amax(1) if bank.native.events.shape[-2] else None   # [keys, P]
    points, against_hits = [], []
    for start, end in occurrences:
        raw = raw_of(start, end)
        present = []
        for i, whole in ws:
            for p, (a, z) in enumerate(events):
                if (a, z) == (start, end) and lit is not None and float(lit[i, p]) >= .5:
                    present.append(whole)
        code = _occurrence_code(basic, identity, raw, present)
        points.append(code if code is not None else basic.codes.new_zeros(basic.width))
    points = torch.stack(points)                                          # [O, width]
    rows = [r for r in bank.rows]
    real = [r for r in rows if r not in basic.place_rows and r not in basic.conjunctions]
    table = read.new_zeros(len(rows), len(brackets), 2)
    if real:
        boundary = basic.read_points(points[None].to(basic.codes), real)  # [2, R, 1, O, 2]
        part, whole = boundary[0, :, 0, :, 0], boundary[1, :, 0, :, 0]   # [R, O]
        against = basic.against_codes[real][:, :basic.identity_width] > .5
        hit = (points[None, :, :basic.identity_width] > .5) & against[:, None]
        added = hit.any(-1).to(part)
        value = torch.stack((part * whole, part * torch.maximum(1 - whole, added)), -1)   # [R, O, 2]
        for k, (a, z) in enumerate(brackets):
            inside = torch.tensor([a <= s and e <= z for s, e in occurrences])
            if bool(inside.any()):
                at = value[:, inside].amax(1)
                for n, r in enumerate(real):
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
