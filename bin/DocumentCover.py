"""Document covers in exact batch streams; membership is an ordered segment.

Only the presentation cursor remembers which document is open. Durable reads
use the retained cover, stream and RowTime columns, never source provenance.
"""
from dataclasses import replace

import torch


def segments(store, stream=None):
    """Map retained rows to their cover occurrences, one per witnessing stream."""
    from RowTime import encoder
    result = {}
    if 'cover' not in store.KINDS or not len(store):
        return result
    times = encoder().decode_index(store.when[:len(store)]).tolist()
    streams = range(store.stream.shape[1]) if stream is None else (int(stream),)
    cover_kind = store.KINDS.index('cover')
    for batch in streams:
        current = None
        for row in sorted(store.stream_rows(batch).tolist(), key=lambda i: (times[i], i)):
            if int(store.record_kind[row]) == cover_kind:
                current = store.occurrence_of(row)
            if current is not None:
                result.setdefault(row, set()).add(current)
    return result


def rows(store, cover):
    return tuple(row for row, covers in segments(store).items() if cover in covers)


def latest(store, stream):
    membership = segments(store, stream)
    return next(reversed(membership.values())).copy().pop() if membership else None


@torch.no_grad()
def open_cover(owner, store, *, stream, document):
    """Reuse the book's concept; write its cover in this presentation."""
    from Spaces import _concept_alloc_of
    from Meaning import ConceptualMeaning
    from Occurrence import document_digest
    key = document_digest(document).hex()
    symbol = store._cover_symbols.get(key)
    alloc = _concept_alloc_of(owner)
    value = None
    if symbol is not None:
        seat = owner._csw_row_of(symbol)
        if seat is None or symbol in alloc.retired:
            raise RuntimeError('a mapped cover concept is unavailable')
        value = owner.similarity_codebook.W[seat]
    meaning = None if value is None else replace(
        ConceptualMeaning.from_description(value[:store.nDim]), role_refs=(('sym', symbol), None, None))
    previous = None if meaning is None else store._address_for_write(meaning, kind='cover',
        document_key=('cover', store.address_domain, symbol),
        content_key=document_digest(('cover', symbol)))[-1]
    if previous is None and len(store) >= store.capacity:
        raise OverflowError('LTM capacity exhausted before document cover')
    if symbol is None:
        owner._preflight_concept_row(1)
        owner._preflight_concept_allocation(1, context='document cover')
        symbol = owner.new_concept()
        alloc.reference_orders[symbol] = 1
        alloc.symbol_types.setdefault(symbol, set()).update(('concept', 'object'))
        alloc.individuals.add(symbol)
        seat = owner._csw_concept_row(1, symbol)
        if seat is None:
            raise RuntimeError('document cover inventory changed after preflight')
        value = owner.similarity_codebook.W[seat]
        generator = store._cover_generator
        generator.set_state(store.cover_rng_state.cpu())
        value.copy_(torch.rand(value.shape, dtype=value.dtype, device='cpu', generator=generator))
        store.cover_rng_state.copy_(generator.get_state())
        derived = getattr(owner.similarity_codebook, 'mereology', None)
        if derived is not None:
            value[derived.percept_width:derived.percept_event_width] = 0.
        alloc.settle(symbol)
        store._cover_symbols[key] = symbol
    meaning = replace(ConceptualMeaning.from_description(value[:store.nDim]),
                      role_refs=(('sym', symbol), None, None))
    row = store.append_meaning(meaning, kind='cover', order=1, stream=int(stream),
        evidence=(1., 0.), document_key=('cover', store.address_domain, symbol),
        content_key=document_digest(('cover', symbol)))
    return store.occurrence_of(row)


def begin_sentence(model, slot, present):
    """Cursor boundary before the paired trials; never called by preview."""
    from Occurrence import source_at
    owner = model._concept_owner()
    store = owner._closed_clause_store()
    if store is None:
        return
    cursor = model.__dict__.setdefault('_document_covers', {})
    predictor = getattr(getattr(model, 'symbolSpace', None), 'expectation', None)
    for batch, active in enumerate(present.tolist()):
        if not active:
            continue
        document, position = source_at(model, batch, slot)
        previous = cursor.get(batch)
        if (previous is None or previous[0] != document or position <= previous[1]
                or store.location(previous[2]) is None):
            if previous is not None:
                owner.mtm.end_document(previous[2], store=store)
            cover = open_cover(owner, store, stream=batch, document=document)
            if predictor is not None:
                predictor.include_occurrence(batch, cover)
        else:
            cover = previous[2]
        cursor[batch] = document, position, cover


def end_document(model, stream):
    owner = model._concept_owner()
    store = owner._closed_clause_store()
    cursor = model.__dict__.get('_document_covers', {})
    previous = cursor.pop(int(stream), None)
    if store is not None:
        cover = previous[2] if previous is not None else latest(store, stream)
        owner.mtm.end_document(cover, store=store)


def read(store, description, *, references=(), max_records, work, role_mask=None):
    """A cover concept cues its bounded segment, without an invented name."""
    wanted = set(references) | {ref for ref in description.role_refs if ref is not None}
    if not wanted:
        return None
    if type(max_records) is not int or max_records < 0:
        raise ValueError('cover read limit must be a nonnegative integer')
    selected, cover_refs = set(), set()
    for reference in wanted:
        if not isinstance(reference, tuple):
            continue
        row = store.location(reference) if reference[:1] == ('ltm',) else None
        candidates = () if row is None else (row,)
        if reference[:1] == ('sym',) and store._index_code_row is not None:
            code = store._index_code_row(reference)
            if code is not None:
                candidates = store.rows_for_code(code, role=0)
        for row in candidates:
            if int(store.record_kind[row]) == store.KINDS.index('cover'):
                selected.add(store.occurrence_of(row))
                cover_refs.add(reference)
    if not selected:
        return None
    # The cover narrows the region. Other occupied roles still constrain
    # the query; a document cue cannot make an unrelated proposition true.
    occupied = description.role_mask if role_mask is None else role_mask
    mask = occupied & occupied.new_tensor([ref not in cover_refs for ref in description.role_refs])
    result, scanned, incomplete = [], 0, []
    for row, covers in segments(store).items():
        if not covers.intersection(selected) or store.occurrence_of(row) in selected:
            continue
        if scanned >= max_records or not work.consume('record'):
            incomplete.append('cover_read_limit')
            break
        scanned += 1
        record = store.row(row)
        if record['kind'] in ('cover', 'estimate', 'question'):
            continue
        meaning = record['meaning']
        if (meaning is None or bool((mask & ~meaning.role_mask).any())
                or not store._scope_contains(description.scope, meaning.scope)
                or not store._scope_contains(description.bindings, meaning.bindings)):
            continue
        similarity = torch.nn.functional.cosine_similarity(description.roles.to(meaning.roles),
                                                           meaning.roles, dim=-1)
        match = float(similarity[mask].clamp(0, 1).mean()) if bool(mask.any()) else 1.
        result.append(dict(record, index=row, match=match))
    return dict(value=tuple(result), records_scanned=scanned, incomplete=tuple(incomplete))
