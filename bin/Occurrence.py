"""Stable source addresses and the separate staging of model-time coordinates."""
import hashlib
import json

import torch


def canonical(value):
    if isinstance(value, bytes):
        return {'bytes': value.hex()}
    if isinstance(value, (tuple, list)):
        return [canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): canonical(item) for key, item in sorted(value.items())}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError('a document key must be stable source data')


def document_digest(key):
    return hashlib.sha256(json.dumps(canonical(key), sort_keys=True,
        separators=(',', ':'), ensure_ascii=True).encode()).digest()


def sentence_key(words):
    """Identified words in order, independent of reading, document and store."""
    digest = hashlib.sha256(b'sentence-identity-v1\0')
    for word in words:
        if not isinstance(word, bytes):
            raise TypeError('sentence identity requires identified word bytes')
        digest.update(len(word).to_bytes(8, 'little'))
        digest.update(word)
    return digest.digest()


def slot_key(address, role):
    """The int64 carrier of a row-address/role reference, never an allocation."""
    if type(address) is not int or address < 0 or role not in (0, 1, 2):
        raise ValueError('a slot requires an occurrence address and canonical role')
    payload = b'occurrence-slot-v1\0' + address.to_bytes(8, 'little', signed=True) + bytes((role,))
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') | (1 << 63)
    value -= 1 << 64
    from RowTime import HORIZON
    if value >= -HORIZON-2:
        raise ValueError('slot hash collides with a reserved occurrence or null')
    return value


def prepared_input(space, raw, result):
    """Keep source coordinates on the prepared value, not a mutable cursor.

    Explicit runEpoch provenance remains authoritative. This also covers raw
    forward/evaluation callers that retain a prepared tensor across epochs.
    Reject stale cursor metadata when prepInput receives unrelated text.
    """
    import copy
    rows = getattr(space, '_prepared_source_rows', None)
    split = getattr(space, '_prepared_source_split', 'train')
    cursor = getattr(space.data, '_address_cursor', None)
    valid = split == 'runtime'
    if rows is not None and cursor is not None and split != 'runtime':
        valid = len(rows) == len(raw)
        for value, source in zip(raw, rows):
            source = source[0] if isinstance(source, (tuple, list)) and len(source) == 1 else source
            if not isinstance(source, int) or not 0 <= source < len(cursor.inputs):
                valid = False
                break
            original = cursor.inputs[source]
            def tensor(item):
                return (space.data.stringTensor(item) if isinstance(item, str)
                        else torch.as_tensor(item)).detach().to('cpu').reshape(-1)
            if not torch.equal(tensor(value), tensor(original)):
                valid = False
                break
    if torch.is_tensor(result):
        result._sentence_source = (split, copy.deepcopy(rows)) if valid and rows is not None else None
    return result


def stage_sources(model, split, source_rows, inputs, document_keys=None):
    """Source coordinates for each packed sentence, before any learned reading."""
    model._source_request = (split, source_rows, document_keys)
    data = getattr(getattr(model, 'inputSpace', None), 'data', None)
    addresses = getattr(data, 'source_addresses', {}).get(str(split), ())
    batch = int(inputs.shape[0]) if torch.is_tensor(inputs) else len(inputs)
    rows = [] if source_rows is None else list(source_rows)
    nested = rows if rows and isinstance(rows[0], (list, tuple)) else [[r] for r in rows]
    manifest = getattr(data, 'source_manifest', None) or {}
    dataset = (manifest.get('dataset', 'direct'), manifest.get('content_sha256', manifest.get('sha256')),
               tuple(item.get('sha256') for item in manifest.get('shards', ()) if isinstance(item, dict)))
    result = []
    for b in range(batch):
        row = []
        for source in nested[b] if b < len(nested) else ():
            if source is None or not 0 <= int(source) < len(addresses):
                continue
            entry = addresses[int(source)]
            document = (document_keys[b] if document_keys is not None else
                        entry.get('document_key', ('dataset', dataset, entry['document'])))
            row.append((document, int(entry.get('sentence', 0)) + 1))
        if not row:
            # Direct callers may supply conversation/turn identity. Otherwise
            # the direct input is its own immutable document; presenting it
            # twice is explicitly a re-reading, not a fabricated new episode.
            value = inputs[b]
            payload = (value.detach().to('cpu').contiguous().numpy().tobytes()
                       if torch.is_tensor(value) else str(value).encode())
            document = (document_keys[b] if document_keys is not None else
                        ('direct-input', hashlib.sha256(payload).hexdigest()))
            row.append((document, 1))
        result.append(tuple(row))
    model._sentence_sources = tuple(result)
    model._sentence_ticks = {}
    stage_position(model, 0, batch, device=inputs.device if torch.is_tensor(inputs) else 'cpu')
    memory = getattr(getattr(model, 'symbolSpace', None), 'what_memory', None)
    if memory is not None:
        memory._address_sources = {b: (('thought', row[0][0], row[0][1]), 0)
                                   for b, row in enumerate(result) if row}


def restage_presented(model, inputs):
    """Update direct-input provenance after omission, before any sentence read."""
    split, rows, documents = model._source_request
    stage_sources(model, split, rows, inputs, documents)


def address_at(model, batch, slot):
    """Stable document identity; it is not an experience's model-time stamp."""
    rows = getattr(model, '_sentence_sources', ())
    if batch < len(rows) and rows[batch]:
        row = rows[batch]
        if slot < len(row):
            return row[slot]
        # A single supplied document may contain several parsed sentences.
        return row[0][0], row[0][1] + slot
    return ('direct-model', batch), slot + 1


def source_at(model, batch, slot):
    """Document and the model tick of its current sentence presentation."""
    return address_at(model, batch, slot)[0], model._sentence_ticks.get(int(slot), model.present())


def relative_positions(model, sentence_ids):
    positions = torch.zeros_like(sentence_ids)
    for b in range(sentence_ids.shape[0]):
        for slot in range(max(1, int(sentence_ids[b].max()) + 1)):
            positions[b] = torch.where(sentence_ids[b] == slot,
                source_at(model, b, slot)[1], positions[b])
    return positions


def stage_position(model, slot, batch, *, device):
    """Stage the current sentence tick before its word perception begins."""
    position = torch.tensor([source_at(model, b, slot)[1] for b in range(batch)],
                            dtype=torch.long, device=device)
    for space in (getattr(model, 'inputSpace', None), getattr(model, 'perceptualSpace', None),
                  *([getattr(model, "wholeSpace")] if getattr(model, "wholeSpace", None) is not None else []), getattr(model, 'symbolSpace', None),
                  *getattr(model, 'conceptualSpaces', ())):
        if space is not None:
            object.__setattr__(space, '_document_position', position)
    # The words were admitted together before the sentence loop. Stamp the
    # current sentence only now: preceding thoughts may have advanced time.
    isp = getattr(model, 'inputSpace', None)
    from PrependedSymbols import word_view
    bands = word_view(isp, 'symbol_when')
    if slot in model._sentence_ticks and torch.is_tensor(bands):
        selected = (isp._packed_sentence_ids == slot) & (word_view(isp, 'concept_rows') >= 0)
        stamp = model.when_encoding.encode(position)[:, None].to(bands)
        isp._ar_word_symbol_when = torch.where(selected[..., None], stamp, bands)


def event_positions(space, shape, *, device):
    """Sentence model time for generic carriers and native event writers."""
    positions = getattr(space, '_document_position', None)
    if positions is None or positions.shape[0] != shape[0]:
        positions = torch.ones(shape[0], dtype=torch.long, device=device)
    return positions.to(device).reshape(shape[0], *((1,) * (len(shape) - 1))).expand(shape)



def operand_name(time):
    """A working grammar operand tags a row time apart from positive concept IDs."""
    time=int(time)
    from RowTime import HORIZON
    if not 0 <= time <= HORIZON:
        raise ValueError('an occurrence time must fit the declared time ladder')
    return -time-2


def operand_time(name):
    """Decode only the occurrence arm of the temporary scalar reference union."""
    from RowTime import HORIZON
    return -int(name)-2 if -HORIZON-2 <= int(name) < -1 else None
