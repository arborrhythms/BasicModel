"""Content identities and source addresses. No clock, random state or parameters."""
from contextlib import contextmanager
import hashlib
import json

import torch

ADDRESS_DOMAIN = 'address-v1'  # Format tag, shared by every store; not an owner ID.


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


def native_content_key(meaning):
    """Content for non-lexical direct writes and unrecoverable legacy rows.

    Sentence ingestion supplies identified words instead. This fallback never
    consults a timestamp, occurrence ID, or store identity.
    """
    roles = meaning.roles.detach().to(device='cpu', dtype=torch.float32).contiguous()
    return hashlib.sha256(b'native-content-v1\0' + roles.numpy().astype('<f4').tobytes()
        + bytes(meaning.role_mask.to('cpu').tolist())
        + json.dumps(meaning.metadata(), sort_keys=True, separators=(',', ':')).encode()).digest()


def address_key(document, index, content):
    """Full 64-bit digest, carried as the signed bit pattern in int64 columns."""
    if type(index) is not int or index < 0 or len(document) != 32 or len(content) != 32:
        raise ValueError('address requires document/content digests and a nonnegative index')
    payload = b'occurrence-address-v1\0' + document + index.to_bytes(8, 'little') + content
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little', signed=True)
    if value in (-1, 0):
        raise ValueError('address hash collides with a reserved null; change the source key')
    return value


def definition_key(word, obj):
    return address_key(document_digest(('DEF', int(word), int(obj))), 0, sentence_key(()))


def slot_key(address, role):
    """The int64 carrier of a row-address/role reference, never an allocation."""
    if type(address) is not int or address in (-1, 0) or role not in (0, 1, 2):
        raise ValueError('a slot requires an occurrence address and canonical role')
    payload = b'occurrence-slot-v1\0' + address.to_bytes(8, 'little', signed=True) + bytes((role,))
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') | (1 << 63)
    value -= 1 << 64
    if value == -1:
        raise ValueError('slot hash collides with a reserved null')
    return value


class AddressIndex(dict):
    """Address key -> row; accept the typed reference envelope at read seams."""
    @staticmethod
    def key(value):
        if isinstance(value, tuple) and len(value) == 3 and value[:2] == ('ltm', ADDRESS_DOMAIN):
            return value[2]
        return value

    def __getitem__(self, value):
        return super().__getitem__(self.key(value))

    def __setitem__(self, value, row):
        return super().__setitem__(self.key(value), row)

    def __contains__(self, value):
        return super().__contains__(self.key(value))

    def get(self, value, default=None):
        return super().get(self.key(value), default)



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
    stage_position(model, 0, batch, device=inputs.device if torch.is_tensor(inputs) else 'cpu')
    memory = getattr(getattr(model, 'symbolSpace', None), 'what_memory', None)
    if memory is not None:
        memory._address_sources = {b: (('thought', row[0][0], row[0][1]), 0)
                                   for b, row in enumerate(result) if row}


def source_at(model, batch, slot):
    rows = getattr(model, '_sentence_sources', ())
    if batch < len(rows) and rows[batch]:
        row = rows[batch]
        if slot < len(row):
            return row[slot]
        # A single supplied document may contain several parsed sentences.
        return row[0][0], row[0][1] + slot
    return ('direct-model', batch), slot + 1


def relative_positions(model, sentence_ids):
    positions = torch.zeros_like(sentence_ids)
    for b in range(sentence_ids.shape[0]):
        for slot in range(max(1, int(sentence_ids[b].max()) + 1)):
            positions[b] = torch.where(sentence_ids[b] == slot,
                source_at(model, b, slot)[1], positions[b])
    return positions


def stage_position(model, slot, batch, *, device):
    """Eager source staging; compiled percept operations only broadcast tensors."""
    position = torch.tensor([source_at(model, b, slot)[1] for b in range(batch)],
                            dtype=torch.long, device=device)
    for space in (getattr(model, 'inputSpace', None), getattr(model, 'perceptualSpace', None),
                  *getattr(model, 'wholeSpaces', ()), getattr(model, 'symbolSpace', None),
                  *getattr(model, 'conceptualSpaces', ())):
        if space is not None:
            object.__setattr__(space, '_document_position', position)


def event_positions(space, shape, *, device):
    """Relative positions for generic carriers, before word-specific stamping."""
    positions = getattr(space, '_document_position', None)
    if positions is None or positions.shape[0] != shape[0]:
        positions = torch.ones(shape[0], dtype=torch.long, device=device)
    return positions.to(device).reshape(shape[0], *((1,) * (len(shape) - 1))).expand(shape)


class OccurrenceRows:
    address_domain = ADDRESS_DOMAIN

    @contextmanager
    def at_address(self, document, index, content, *, timestamp=None):
        previous = self.__dict__.get('_write_address')
        self._write_address = (document_digest(document), int(index), content, timestamp)
        try:
            yield
        finally:
            self._write_address = previous

    def content_key(self, row):
        return bytes(self.sentence_content_keys[int(row)].tolist())

    def identity_code(self, row, pairs, ones):
        """Read the fixed content direction at its durable definedness."""
        from MeaningCodes import identity_code, defined_code
        direction = identity_code(self.content_key(row), pairs, ones, like=self.slots)
        return defined_code(direction, self.witness_count[int(row)])

    def rows_for_content(self, key):
        return tuple(row for row in range(len(self)) if self.content_key(row) == key)

    def write_timestamp(self, row, timestamp=None):
        source = getattr(self, '_timestamp_source', None)
        if timestamp is None and source is not None:
            timestamp = source()
        if timestamp is None:
            timestamp = float(self._next_ts)
        self.timestamp[row] = float(timestamp)
        self._next_ts.fill_(max(int(self._next_ts), int(timestamp) + 1))

    def _address_for_write(self, meaning, *, kind, document_key=None,
                           sentence_index=0, content_key=None, key=None):
        context = self.__dict__.get('_write_address')
        content = content_key or (context[2] if context else native_content_key(meaning))
        document = (document_digest(document_key) if document_key is not None else
                    context[0] if context else document_digest(('direct', kind)))
        index = context[1] if context and document_key is None else int(sentence_index)
        identity = address_key(document, index, content) if key is None else int(key)
        previous = self._index_occurrences.get(identity)
        if previous is not None and (bytes(self.document_keys[previous].tolist()) != document
                or int(self.sentence_index[previous]) != index or self.content_key(previous) != content):
            raise RuntimeError('64-bit address collision; refusing to merge different occurrences')
        return identity, document, index, content, previous

    def _drop_row_postings(self, row):
        for key, rows in tuple(self._leaf_postings.items()):
            if row in rows:
                rows.remove(row)
            if not rows:
                del self._leaf_postings[key]

    def _estimate_write_address(self, meaning, *, document=None, stream=None):
        context = self.__dict__.get('_write_address')
        source = ('thought', context[0], context[1]) if context else ('thought', document, stream)
        return dict(document_key=source, sentence_index=0, content_key=native_content_key(meaning))

    def remap_legacy_references(self, value):
        """Apply this checkpoint's old handle/address map to any sidecar owner."""
        if isinstance(value, dict):
            return {key: self.remap_legacy_references(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            mapped = getattr(self, '_legacy_reference_map', {}).get(tuple(value)) if (
                len(value) == 3 and isinstance(value[0], str) and value[0] == 'ltm') else None
            if mapped is not None:
                return mapped
            return type(value)(self.remap_legacy_references(item) for item in value)
        return value

    def _resize_checkpoint_rows(self, state, prefix):
        """A larger configured store can load an older, smaller checkpoint."""
        slots = state.get(prefix + 'slots')
        if slots is None or slots.shape[0] == self.capacity:
            return
        count = int(state.get(prefix + 'count', 0))
        if count > self.capacity:
            raise OverflowError('checkpoint occurrences exceed configured LTM capacity')
        old_capacity = slots.shape[0]
        for name, default in self._buffers.items():
            value = state.get(prefix + name)
            if (name.startswith('posting_') or value is None or value.ndim == 0
                    or value.shape[0] != old_capacity or default.ndim == 0
                    or default.shape[0] != self.capacity
                    or value.shape[1:] != default.shape[1:]):
                continue
            grown = default.clone().to(value)
            grown[:count] = value[:count]
            state[prefix + name] = grown
        # Counter columns from pre-address checkpoints are migrated next.
        old = state.get(prefix + 'occurrence_id')
        if old is not None:
            grown = old.new_full((self.capacity,), -1)
            grown[:count] = old[:count]
            state[prefix + 'occurrence_id'] = grown
        if prefix + 'address_keys' in state and count:
            # The address is unchanged; its coordinate carrier uses the new
            # configured period, just as a newly written row would.
            from Spaces import WhenEncoding
            positions = state[prefix + 'sentence_index'][:count]
            encoder = getattr(self, '_address_encoding', None)
            if encoder is None:
                encoder = WhenEncoding(n_when=4).set_capacity(max(self.capacity, int(positions.max()) + 1))
            state[prefix + 'when'][:count] = encoder.encode(positions).to(state[prefix + 'when'])
            state[prefix + 'when'][:count][state[prefix + 'rel_type'][:count] == self.REL_DEF] = 0

    def _migrate_address_tensors(self, state, prefix):
        """Old snapshots lack source coordinates: retain them in a legacy document.

        Their original handles identify positions in that document. We cannot
        infer a corpus document from a clock. New ingestion always supplies its
        real source address; no legacy vector is claimed to recover words.
        """
        old_ids = state.pop(prefix + 'occurrence_id', None)
        old_namespace = state.pop(prefix + '_occurrence_namespace', None)
        state.pop(prefix + '_next_occurrence', None)
        if prefix + 'address_keys' in state:
            self._legacy_reference_map = {}
            return
        from Meaning import ConceptualMeaning
        count = int(state.get(prefix + 'count', self.count))
        old_ids = torch.arange(count) if old_ids is None else old_ids[:count]
        namespace = 'pre-occurrence' if old_namespace is None else bytes(old_namespace.tolist()).hex()
        document = document_digest(('legacy-checkpoint', namespace))
        buffers = {name: getattr(self, name).clone() for name in
            ('address_keys', 'document_keys', 'sentence_content_keys', 'sentence_index', 'witness_count')}
        prior_content = state.get(prefix + 'sentence_content_keys')
        row_ids = state[prefix + 'row_ids'].clone()
        refs = state[prefix + 'refs'].clone()
        mapping, native = {}, {}
        old_to_new = {}
        for row, old in enumerate(old_ids.tolist()):
            meaning = ConceptualMeaning(state[prefix + 'slots'][row], state[prefix + 'role_mask'][row])
            content = (bytes(prior_content[row].tolist()) if prior_content is not None
                       and bool(prior_content[row].any()) else native_content_key(meaning))
            row_document, position = document, int(old)
            if int(state[prefix + 'rel_type'][row]) == self.REL_DEF:
                row_document = document_digest(('DEF', int(refs[row, 0]), int(refs[row, 2])))
                position, content = 0, sentence_key(())
            key = address_key(row_document, position, content)
            buffers['address_keys'][row] = key
            buffers['document_keys'][row] = buffers['document_keys'].new_tensor(list(row_document))
            buffers['sentence_content_keys'][row] = buffers['sentence_content_keys'].new_tensor(list(content))
            buffers['sentence_index'][row], buffers['witness_count'][row] = position, 1
            mapping[('ltm', namespace, int(old))] = ('ltm', ADDRESS_DOMAIN, key)
            old_to_new[int(old)] = key
            previous = int(row_ids[row])
            eternal = (int(state[prefix + 'rel_type'][row]) == self.REL_NONE
                       and previous > 0 and previous == int(refs[row, 0]))
            if not eternal:
                if previous not in (-1, 0):
                    native[previous] = key
                row_ids[row] = key
        self._legacy_reference_map, self._legacy_id_map = mapping, old_to_new
        self._legacy_namespace = namespace
        self._legacy_native_map = native
        self._legacy_fingerprints = state[prefix + 'semantic_fingerprint'].clone()
        for old, new in native.items():
            refs[state[prefix + 'refs'] == old] = new
        state[prefix + 'refs'], state[prefix + 'row_ids'] = refs, row_ids
        from Spaces import WhenEncoding
        encoder = getattr(self, '_address_encoding', None)
        if encoder is None:
            encoder = WhenEncoding(n_when=4).set_capacity(max(self.capacity, max(old_ids.tolist(), default=0) + 1))
        when = state[prefix + 'when'].clone()
        when[:count] = encoder.encode(buffers['sentence_index'][:count]).to(when)
        when[:count][state[prefix + 'rel_type'][:count] == self.REL_DEF] = 0
        state[prefix + 'when'] = when
        state.update({prefix + name: value for name, value in buffers.items()})

    def _migrate_address_extras(self, extras):
        if extras['version'] == 5:
            return extras
        if extras.get('namespace') != getattr(self, '_legacy_namespace', None):
            raise ValueError('legacy semantic checkpoint occurrence namespace differs')
        records = []
        for record in extras.get('records', ()):
            old = int(record['id'])
            new = self._legacy_id_map.get(old)
            if new is None:
                raise ValueError('legacy semantic checkpoint occurrence is unavailable')
            index = self._index_occurrences.get(new)
            if index is None:
                continue  # Existing schema migration deliberately discarded this row.
            fingerprint = self._context_fingerprint(record.get('context'), record.get('text'),
                record.get('expectation'), record.get('definition'))
            if fingerprint != self.semantic_fingerprint[index].tolist():
                raise ValueError('truth semantic content differs from its checkpoint fingerprint')
            migrated = self.remap_legacy_references(record)
            migrated['id'] = new
            migrated.setdefault('expectation', None)
            records.append(migrated)
        result = dict(version=5, namespace=ADDRESS_DOMAIN, records=records)
        # Validate the old binding first; the new binding covers remapped edges.
        for record in records:
            index = self._index_occurrences[record['id']]
            self.semantic_fingerprint[index] = self.semantic_fingerprint.new_tensor(
                self._context_fingerprint(record.get('context'), record.get('text'),
                    record.get('expectation'), record.get('definition')))
        return result
