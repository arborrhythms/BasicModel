"""Compare the batched form table with the per-row read on the §64 source."""
import torch

from test_review13_subspaces import fixture, word


def reference(owner, rows):
    from PerceptCollections import definitions, code
    store = owner._concept_allocator
    book = owner.similarity_codebook
    charts = book.mereology
    result = []
    for row in rows.flatten().tolist():
        output = torch.zeros_like(book.W[row])
        concept = owner.concept_id_at_row(row)
        retained = (concept is not None and concept not in store.retired and
                    ('concept' in store.symbol_types.get(concept, ()) or concept in store.individuals
                     or concept in store.form_regions.values()))
        if owner._row_order(row) == 0:
            edges = {}
            for (source, percept, negated), weight in store.order_zero_members.items():
                if source == row:
                    pair = edges.setdefault(percept, [0., 0.])
                    pair[int(negated)] += weight
            weighted = [(p, f-a) for p, (f, a) in sorted(edges.items()) if f > a]
            if row in definitions(owner):
                output[:charts.percept_width] = code(owner, row)[:charts.percept_width]
            elif retained:
                output[:charts.percept_width] = book.W[row, :charts.percept_width]
            elif weighted:
                points = torch.stack([owner.basic_percepts.code(p) for p, _ in weighted])
                weights = points.new_tensor([w for _, w in weighted])
                output[:charts.percept_width] = (points * weights[:, None]).sum(0) / weights.sum()
            elif concept in owner.basic_percepts.words:
                output[:charts.percept_width] = owner.basic_percepts.code(owner.basic_percepts.words[concept])
        first, n = charts.percept_event_width, charts.meaning_pairs
        output[first:first+n] = charts.positive_codes([row])[0]
        output[first+n:first+2*n] = book.W[row, first+n:first+2*n]
        result.append(output)
    return torch.stack(result).reshape(*rows.shape, -1)


def test_scattered_percepts_match_single_grouped_negative_and_retained_rows():
    from test_form_subspaces import extended_fixture
    owner, book, native, _, alloc, a, b = extended_fixture()
    c = word(owner, [(4, .75), (7, -.5)], (0,))
    owner.add_concept_feature(c, 'bp', owner.basic_percepts.words[owner.concept_id_at_row(b)], .375)
    rows = torch.tensor([[b, a, c, a], [a, b, c, c]])
    def check():
        book.mereology._form_valid.zero_()
        expected = reference(owner, rows)
        actual = book.lookup_rows(rows, streams=True)
        assert torch.equal(actual, expected)
    check()
    with torch.no_grad():
        for key in tuple(alloc.order_zero_members):
            alloc.order_zero_members[key] *= .75
        native.W[4, 1] += .25
    check()
    alloc.individuals.add(owner.concept_id_at_row(a))
    book.W[a, :3] = torch.tensor([.11, .22, .33])
    check()


def test_streamed_reads_share_the_form_cache(monkeypatch):
    owner, book, _, _ = fixture(batch_size=3)
    rows = torch.tensor([word(owner, [(4, 1.)]), word(owner, [(7, .5)]),
                         word(owner, [((4, 7), 1.)])])
    charts = book.mereology
    seen = []
    native = charts.derive
    def observed(indices, **kwargs):
        seen.append(indices.shape)
        return native(indices, **kwargs)
    monkeypatch.setattr(charts, 'derive', observed)
    expected = reference(owner, rows)
    charts._form_valid.zero_()
    initial = charts.form_rows_derived
    result = book.lookup_rows(rows[None].expand(3, -1), streams=True)
    assert torch.equal(result, expected.expand_as(result))
    assert seen == [torch.Size((3, 3))] and charts.form_rows_derived-initial == 3
    assert torch.equal(charts.derive(rows), expected)
    assert charts.form_rows_derived-initial == 3
