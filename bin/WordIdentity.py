"""Rung-zero identity: a form's code is the set of its parts.

A part is a run of adjacent characters. A form's parts are its characters,
the space among them, and every longer part that has been added and lies
inside it. A longer part is added only where two forms would otherwise
have the same parts, or where it is taught (``admit_part``); once added it
is carried by every form that holds that run. So one form lies inside
another exactly when its set of parts lies inside the other's. Atom and
word rows belong to the existing RadixLayer; their codes are fixed
functions of the parts' bytes, not parameters.
"""
import hashlib
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache

import numpy as np
import torch
from torch import nn


def resolve_identity_layout(config):
    """Resolve text form storage before constructing any spaces.

    A form's code is a parts half and a wholes half (sigma-pi rule 50):
    ``identityPairDim`` coordinates for the parts, spread over all of them,
    and ``propertyDim`` for the wholes. Preserve the declared meaning
    complement and live slot/row counts; numeric configurations are
    untouched.
    """
    from architecture import canonical_shape
    arch = config.get('architecture', {})
    if arch.get('data', {}).get('dataType', arch.get('dataType')) != 'embedding':
        return
    ps = config.setdefault('PartSpace', {})
    if not ps.get('identityByParts', True):
        return
    if 'identityLengthDim' in ps:
        raise ValueError('identityLengthDim is retired: the parts half has no block of 32')
    pair = int(ps.setdefault('identityPairDim', 64))
    ones = int(ps.setdefault('identityOnes', 3))
    property_width = int(ps.setdefault('propertyDim', 64))
    property_ones = int(ps.setdefault('propertyOnes', 3))
    if pair < ones or ones < 1:
        raise ValueError('identity requires D >= s > 0')
    if not 0 < property_ones <= property_width:
        raise ValueError('property codes require propertyDim >= propertyOnes > 0')
    width = pair + property_width
    old_ps = int(ps.get('nDim', 0))
    old_cs = int(config.get('ConceptualSpace', {}).get('nDim', old_ps)) or old_ps
    form_event = width + sum(canonical_shape('PartSpace'))
    complement = max(0, old_cs - old_ps)
    # Recurrent WS and the output head consume CS events, not their own
    # native form width. Retain deliberately inconsistent declarations so
    # normal configuration validation still rejects them.
    for section in ('WholeSpace', 'OutputSpace'):
        item = config.get(section, {})
        if int(item.get('nInputDim', 0) or 0) == old_cs and old_cs > 0:
            item['nInputDim'] = form_event + complement
    for section, event in (('InputSpace', form_event), ('PartSpace', form_event),
                           ('WholeSpace', form_event),
                           ('ConceptualSpace', form_event + complement)):
        item = config.setdefault(section, {})
        old = int(item.get('nDim', 0))
        item['nDim'] = event
        for key in ('nInputDim', 'nOutputDim'):
            if int(item.get(key, 0) or 0) == old and old > 0:
                item[key] = event


def text_of(raw):
    return bytes(raw).decode('utf8', 'surrogateescape')


def bytes_of(text):
    return text.encode('utf8', 'surrogateescape')


@lru_cache(maxsize=16384)
def bit_stream(atom, dimension):
    seed = int.from_bytes(hashlib.sha256(atom).digest()[:8], 'little')
    return tuple(map(int, np.random.default_rng(seed).permutation(dimension)))


def base_atoms(raw):
    """A form's characters, each once, in order; the space is one of them."""
    return tuple(dict.fromkeys(('part', ch) for ch in text_of(raw)))


def runs(text, size):
    return {text[i:i+size] for i in range(len(text)-size+1)}


def distinguishing_run(left, right, *, one_sided=False):
    """The shortest run that ``left`` holds and ``right`` does not, or, by
    default, that one of the two holds and the other does not. Of runs of
    one length the first in code-point order is taken; ``None`` if none."""
    a, b = text_of(left), text_of(right)
    for size in range(1, max(len(a), len(b))+1):
        held, other = runs(a, size), runs(b, size)
        found = held - other if one_sided else held ^ other
        if found:
            return min(found)
    return None


class WordIdentity(nn.Module):
    PREFIX = b'\x00identity/atom/'
    #: Forms of the repair's queue tested against every stored code at once.
    TESTED_AT_ONCE = 128
    WORD_PREFIX = b'\x00identity/form/'

    def __init__(self, store, pair_dim=64, ones=3, property_width=0, property_ones=3, basic_capacity=None):
        super().__init__()
        object.__setattr__(self, 'store', store)
        self.pair_dim, self.ones = int(pair_dim), int(ones)
        self.width = self.pair_dim
        self.property_width, self.property_ones = int(property_width), int(property_ones)
        if (store.dim != self.width + self.property_width or self.property_width < 0
                or not 0 < self.ones <= self.pair_dim):
            raise ValueError('Radix bank width must be identity + property, with 0 < s <= D')
        object.__setattr__(self, 'properties', None)
        self._property_revision = None
        self._fixed_bytes = {}
        self._codes = {}
        self.atoms, self.bits, self.words, self.word_rows, self.forms = {}, {}, {}, {}, {}
        self.word_keys = {}
        # Each stored form's links, written when it is admitted and saved:
        # the stored forms that hold it as a run (its holders, its wholes
        # among the forms) and those it holds as a run (its parts among the
        # forms). A form that holds another's characters and not the run is
        # neither.
        self.holders = {}
        self.inside = {}
        # No code lies inside another's by accident (rule 61). A part may be
        # given coordinates of its own over both halves; each stored form's
        # code is kept packed, one row per form in the order of admission,
        # so that a new code is tested against every stored code at once.
        self.total = self.width + self.property_width
        self.limbs = (self.total + 63) // 64
        self.given = {}
        self.order = {}
        self._ordered = []
        self._packed = torch.zeros(0, self.limbs, dtype=torch.long, device='cpu')
        self.usage = torch.zeros(self.total, dtype=torch.long, device='cpu')
        self._given_revision = 0
        self._parts_key = {}
        # The earlier forms whose code the last admission changed, and those
        # it gave a coordinate, whether or not they had it.
        self.last_changed = self.last_touched = frozenset()
        self._altered = set()
        self.failures = []
        self._table_stamp = self._table = None
        self._atom_rows = {}
        self._catalog = {}
        # Each added part, with the two forms it told apart (None if taught).
        self.added = {}
        # Each parts half's stored forms, for a reader that has only that.
        self.by_parts = {}
        self.word_witnesses = {}
        self.nested = {}
        self._marker_pattern = None
        self.revision = 0
        from BasicPercepts import BasicPercepts
        self.basic = BasicPercepts(store, self.width, self.property_width,
            store._capacity if basic_capacity is None else basic_capacity)
        basis = store._basis
        if not hasattr(basis, 'fixed_row_mask'):
            basis.register_buffer('fixed_row_mask', torch.zeros(store._capacity, dtype=torch.bool, device=basis.W.device))
            basis.register_buffer('fixed_row_codes', torch.zeros_like(basis.W, requires_grad=False))

    def atom_code(self, atom, bits=None):
        """A part's fixed bits, drawn over every coordinate of the parts half."""
        if bits is None and atom in self._codes:
            return self._codes[atom]
        _kind, value = atom
        n = self.bits.get(atom, self.ones) if bits is None else int(bits)
        result = torch.zeros(self.width, device='cpu')
        result[list(bit_stream(bytes_of(value), self.pair_dim)[:n])] = 1
        given = self.given.get(atom)
        if given is not None:
            result = torch.maximum(result, given[:self.width])
        if bits is None:
            self._codes[atom] = result
        return result

    def catalog_code(self, atom, bits=None):
        """A part's code in the wholes half: the catalog drawn a second time
        in that half's own coordinates. A character's is its byte's in the
        table of wholes; a part given more bits in the parts half is given
        as many more here (``bits``: read at those bits). The coordinates
        given to a part (rule 61) are joined in either half."""
        bits = self.bits.get(atom, self.ones) if bits is None else int(bits)
        n = self.property_ones + bits - self.ones
        key = (atom, n)
        if key not in self._catalog:
            from PerceptProperties import catalog_coordinates
            result = torch.zeros(self.property_width, device='cpu')
            result[list(catalog_coordinates(bytes_of(atom[1]), self.property_width, n))] = 1
            self._catalog[key] = result
        given = self.given.get(atom)
        if given is None:
            return self._catalog[key]
        return torch.maximum(self._catalog[key], given[self.width:])

    def tightest(self, rows):
        """The tightest whole of the form whose parts are these store rows:
        the OR of their catalog codes in the wholes half. ``None`` where a
        row is no part or form of this catalog."""
        if not self.property_width:
            return None
        key = (tuple(map(int, rows)), self.revision, self._given_revision, len(self.added),
               len(self.atoms), len(self.word_rows))
        if key in self._catalog:
            return self._catalog[key]
        if len(self._atom_rows) != len(self.atoms) + len(self.word_rows):
            self._atom_rows = {row: (atom,) for atom, row in self.atoms.items()}
            self._atom_rows.update({row: raw for raw, row in self.word_rows.items()})
        codes = []
        for row in map(int, rows):
            held = self._atom_rows.get(row)
            if held is None:
                return None
            atoms = held if isinstance(held, tuple) else self.part_atoms(held)
            codes.extend(self.catalog_code(atom) for atom in atoms)
        result = torch.stack(codes).amax(0) if codes else None
        self._catalog[key] = result
        return result

    def occurrence_bank(self, device):
        """Group native part references by spelling length for the occurrence read.

        The cache holds addresses and fixed catalog coordinates. Native part
        codes are read through the actual basis on each differentiable forward.
        """
        stamp = (len(self.atoms), len(self.added), self.revision, self._given_revision, device)
        cached = self._catalog.get('occurrence_bank')
        if cached is not None and cached[0] == stamp:
            return cached[1]
        groups = {}
        for atom, row in self.atoms.items():
            text = atom[1]
            if atom[0] != 'part' or (len(text) != 1 and text not in self.added):
                continue
            try:
                raw = text.encode('utf-8')
            except UnicodeEncodeError:
                # A surrogateescape byte may be part of a valid character in
                # another span. Its interpretation depends on the span's edges.
                self._catalog['occurrence_bank'] = stamp, None
                return None
            groups.setdefault(len(raw), []).append((raw, row, atom))
        spellings, rows, catalogs = [], [], []
        for size in sorted(groups):
            entries = groups[size]
            spellings.append(torch.tensor([list(raw) for raw, _, _ in entries],
                                          dtype=torch.uint8, device=device))
            rows.extend(row for _, row, _ in entries)
            if self.property_width:
                catalogs.extend(self.catalog_code(atom) for _, _, atom in entries)
        addresses = torch.tensor(rows, dtype=torch.long, device=device)
        catalog = (torch.stack(catalogs).to(device) if catalogs else
                   torch.empty(len(rows), self.property_width, device=device))
        bank = tuple(spellings), addresses, catalog
        self._catalog['occurrence_bank'] = stamp, bank
        return bank

    def _link(self, raw):
        """Link a new stored form to the stored forms it holds as a run (its
        parts among the forms) and to those that hold it (its holders), on
        both sides of each link."""
        text = bytes(raw)
        self.inside[text] = {text[start:start + size] for size in range(1, len(text))
                             for start in range(len(text) - size + 1)
                             if text[start:start + size] in self.inside}
        # A stored form that holds this one holds each of its parts and each
        # of its characters, so its holders are among the holders of any of
        # them: the shortest such list is read, and each candidate is
        # checked for the run.
        by_byte = self._forms_by_byte(text)
        lists = [self.holders[run] for run in self.inside[text]]
        lists.extend(by_byte[byte] for byte in set(text))
        shortest = min(lists, key=len)
        self.holders[text] = {word for word in shortest if word != text and text in word}
        for run in self.inside[text]:
            self.holders[run].add(text)
        for holder in self.holders[text]:
            self.inside[holder].add(text)

    def _forms_by_byte(self, new):
        """The stored forms by each character they hold, with ``new`` (a
        form just stored) among them; rebuilt when it does not follow the
        stored forms (after a load)."""
        index = self.__dict__.get('_by_byte')
        if index is None or self.__dict__.get('_by_byte_count') != len(self.words) - 1:
            index = {}
            for word in self.words:
                if word != new:
                    for byte in set(word):
                        index.setdefault(byte, set()).add(word)
        for byte in set(new):
            index.setdefault(byte, set()).add(new)
        self.__dict__['_by_byte'] = index
        self.__dict__['_by_byte_count'] = len(self.words)
        return index

    def nearest_holders(self, raw):
        """The nearest of a form's holders: those that add the fewest
        characters to it."""
        holders = self.holders.get(bytes(raw), ())
        if not holders:
            return ()
        least = min(map(len, holders))
        return tuple(sorted(h for h in holders if len(h) == least))

    def against_atoms(self, raw):
        """The parts that a form's nearest holders add to it."""
        own = set(self.words.get(bytes(raw), ()))
        # A part whose run holds the form is found only where the characters
        # its holder adds are found, so it adds nothing to them.
        text = text_of(raw)
        return tuple(sorted({atom for holder in self.nearest_holders(raw)
                             for atom in self.words[holder] if atom not in own and text not in atom[1]}))

    def atoms_of(self, raw):
        """The parts of a form: its characters and the added parts inside it
        (a stored form being one), in the order the parts were added."""
        text = text_of(raw)
        added = self.added
        n = len(text)
        if n * (n + 1) // 2 < len(added):
            rank = self._added_rank()
            found = {text[i:j] for i in range(n) for j in range(i + 1, n + 1)}.intersection(added)
            inside = sorted(found, key=rank.__getitem__)
        else:
            inside = [run for run in added if run in text]
        return base_atoms(raw) + tuple(('part', run) for run in inside)

    def _added_rank(self):
        """Each added part's place in the order of addition."""
        stamp = (id(self.added), len(self.added))
        if self.__dict__.get('_rank_stamp') != stamp:
            self.__dict__['_rank'] = {run: i for i, run in enumerate(self.added)}
            self.__dict__['_rank_stamp'] = stamp
        return self.__dict__['_rank']

    @torch.no_grad()
    def _fixed(self, row, value, raw=None):
        basis = self.store._basis
        value = value.to(basis.W)
        if raw is not None:
            self._fixed_bytes[int(row)] = bytes(raw)
        if self.property_width:
            value = torch.nn.functional.pad(value[:self.width], (0, self.property_width))
        basis.fixed_row_mask[row] = True
        basis.fixed_row_codes[row].copy_(value)
        basis.W[row].copy_(value)

    @torch.no_grad()
    def refresh_property_fills(self, properties=None):
        if properties is not None:
            if properties.property_width != self.property_width:
                raise ValueError('identity and property layout widths differ')
            object.__setattr__(self, 'properties', properties)
        if not self.property_width or self.properties is None:
            return
        # BasicPercepts reads the property lanes live; identity atoms own no fill.
        self._property_revision = self.properties.revision

    def _storage_code(self, identity_code):
        """Admission reserves both subspaces; _fixed writes the property fill."""
        return torch.nn.functional.pad(identity_code, (0, self.property_width))

    def _atom(self, atom, bits=None):
        if bits is not None:
            self.bits[atom] = max(self.bits.get(atom, self.ones), int(bits))
            self._codes.pop(atom, None)
            self._catalog.pop('occurrence_bank', None)
        code = self.atom_code(atom)
        if atom not in self.atoms:
            key = self.PREFIX + atom[0].encode() + b'/' + bytes_of(atom[1])
            self.atoms[atom] = self.store.insert(key, init_vector=self._storage_code(code))
        self._fixed(self.atoms[atom], code, bytes_of(atom[1]))
        return self.atoms[atom]

    def form(self, raw, *, override=None):
        """The fold of a form's parts' codes in the fixed width.

        ``override`` is ``(atom, bits)``: that part read at those bits.
        """
        raw = bytes(raw)
        atoms = self.words.get(raw) or self.atoms_of(raw)
        codes = [self.atom_code(a, override[1] if override is not None and a == override[0] else None)
                 for a in atoms]
        return torch.stack(codes).amax(0)

    def code(self, raw, *, override=None):
        """A form's code as rule 56 compares it: its parts half (:meth:`form`)
        and its own tightest whole, the OR of its parts' catalog codes in the
        wholes half. Both are read from its parts, so a set of parts found
        gives the same code (:meth:`spelling`). ``override`` as in
        :meth:`form`."""
        form = self.form(raw, override=override)
        if not self.property_width:
            return form
        atoms = self.words.get(bytes(raw)) or self.atoms_of(raw)
        wholes = torch.stack([self.catalog_code(a, override[1] if override is not None and a == override[0]
                                                else None) for a in atoms]).amax(0)
        return torch.cat((form, wholes))

    @staticmethod
    def key(form):
        return np.packbits(form.detach().eq(1).cpu().numpy(), bitorder='little').tobytes()

    def _keyed(self, word, key, parts_key):
        """Record a form's key and its parts half's key."""
        self.forms[key] = word
        self.word_keys[word] = key
        self.by_parts.setdefault(parts_key, set()).add(word)
        self._parts_key[word] = parts_key

    def _reindex(self):
        """Key every stored form and pack its code, in the order of
        admission. Returns the first two forms found with one code, if any."""
        self.forms, self.word_keys, self.by_parts, self._parts_key = {}, {}, {}, {}
        self.order = {word: index for index, word in enumerate(self.words)}
        self._ordered = list(self.words)
        codes = [self.code(raw) for raw in self._ordered]
        self._packed = (self._pack(torch.stack(codes)) if codes
                        else torch.zeros(0, self.limbs, dtype=torch.long, device='cpu'))
        collision = None
        for raw, code in zip(self._ordered, codes):
            key = self.key(code)
            if key in self.forms and self.forms[key] != raw:
                collision = collision or (self.forms[key], raw)
                continue
            self._keyed(raw, key, self.key(code[:self.width]))
        return collision

    def _unkey(self, word):
        old = self.word_keys.pop(word, None)
        if old is not None and self.forms.get(old) == word:
            del self.forms[old]
            self.store.hash_map.pop(self.WORD_PREFIX+old, None)
            self.store.radix_trie.remove(self.WORD_PREFIX+old)
        parts_key = self._parts_key.pop(word, None)
        held = self.by_parts.get(parts_key)
        if held is not None:
            held.discard(word)
            if not held:
                del self.by_parts[parts_key]

    # Packed codes ---------------------------------------------------------

    def _pack(self, codes):
        """Codes of zeros and ones, ``[n, width + property_width]``, as rows
        of 64-bit words, ``[n, limbs]``: coordinate ``i`` is bit ``i % 64``
        of word ``i // 64``."""
        bits = codes.detach().cpu().reshape(-1, self.total).eq(1).numpy()
        packed = np.packbits(bits, axis=-1, bitorder='little')
        packed = np.pad(packed, ((0, 0), (0, 8 * self.limbs - packed.shape[-1])))
        return torch.from_numpy(np.ascontiguousarray(packed).view('<i8').copy())

    def _unpack(self, packed):
        """Rows of 64-bit words as codes of zeros and ones (``long``)."""
        bits = (packed[..., None] >> torch.arange(64, dtype=torch.long, device='cpu')) & 1
        return bits.flatten(-2)[..., :self.total]

    def _place(self, raw):
        """A new stored form's row of the packed codes."""
        index = len(self._ordered)
        self.order[raw] = index
        self._ordered.append(raw)
        if index >= len(self._packed):
            grown = torch.zeros(max(1024, 2 * len(self._packed)), self.limbs, dtype=torch.long, device='cpu')
            grown[:len(self._packed)] = self._packed
            self._packed = grown
        self._packed[index] = self._pack(self.code(raw))[0]

    def packed_code(self, raw):
        """A stored form's code as the packed rows hold it."""
        return self._unpack(self._packed[self.order[bytes(raw)]]).to(torch.float32)

    @torch.no_grad()
    def _rewrite(self, forms):
        """Write the store rows and the keys of ``forms`` from their packed
        codes, the rows in one indexed write. Returns the pairs of forms
        that now have one code."""
        forms = sorted(forms, key=self.order.__getitem__)
        if not forms:
            return []
        packed = self._packed[torch.tensor([self.order[word] for word in forms], device='cpu')]
        bits = self._unpack(packed)
        basis = self.store._basis
        rows = torch.tensor([self.word_rows[word] for word in forms], device=basis.W.device)
        value = torch.nn.functional.pad(bits[:, :self.width].to(basis.W), (0, self.property_width))
        basis.fixed_row_mask[rows] = True
        basis.fixed_row_codes[rows] = value
        basis.W[rows] = value
        keys = packed.numpy().view(np.uint8)[:, :(self.total + 7) // 8]
        parts = np.packbits(bits[:, :self.width].numpy().astype(bool), axis=-1, bitorder='little')
        collided = []
        for word, key, parts_key in zip(forms, keys, parts):
            key = key.tobytes()
            if self.word_keys.get(word) != key:
                self._unkey(word)
                other = self.forms.get(key)
                if other is not None and other != word:
                    collided.append((other, word))
                self._keyed(word, key, parts_key.tobytes())
            row = self.word_rows[word]
            self.store.hash_map[self.WORD_PREFIX + key] = row
            self.store.radix_trie.insert(self.WORD_PREFIX + key, row)
        return collided

    # Rule 56: two forms with one code ------------------------------------

    def _next_part(self, left, right):
        """Rule 43's next part for two forms with one code (rule 56): the
        shortest run that one holds and the other does not and that is not
        a part yet; of runs of one length the first in code-point order."""
        a, b = text_of(left), text_of(right)
        known = {atom[1] for atom in self.atoms}
        for size in range(1, max(len(a), len(b)) + 1):
            found = (runs(a, size) ^ runs(b, size)) - known
            if found:
                return min(found)
        return None

    def _add_part(self, run, between):
        self.added[run] = between
        atom = ('part', run)
        self._atom(atom)
        for word in self.words:
            if run in text_of(word):
                self.words[word] = self.atoms_of(word)
        return atom

    def _holders(self, atoms):
        atoms = set(atoms)
        return [word for word, parts in self.words.items() if atoms.intersection(parts)]

    def _seen(self, word, *, override=None):
        """A stored form's code as its basic percept keeps it: the wholes
        half within the classes that hold every one of its characters
        (:meth:`_kept`)."""
        code = self.code(word, override=override)
        return code * self._kept(word).to(code)

    def _twin(self, word, code):
        """A stored form with the same parts half whose code, as its basic
        percept keeps it, is the same as ``word``'s, or ``None``."""
        if not self.property_width:
            return None
        held = self.by_parts.get(self.key(code[:self.width]), ())
        if not held:
            return None
        mine = self.key(self._seen(word))
        for other in sorted(held, key=self.order.__getitem__):
            if other != word and self.key(self._seen(other)) == mine:
                return other
        return None

    def _separate(self, left, right, *, read=False):
        """Tell two forms with one code apart (rule 56); return the forms
        whose code changed, or ``None`` where no part tells them apart.

        Rule 43's next part is added, the shortest run that one holds and
        the other does not, until their codes differ. Where no such run is
        left, an added part that one holds and the other does not is given
        more bits, the part held by the fewest forms first.
        """
        changed = []
        seen = self._seen if read else (lambda word, **kw: self.code(word, **kw))
        alike = lambda: self.key(seen(left)) == self.key(seen(right))
        while alike():
            run = self._next_part(left, right)
            if run is None:
                break
            changed.append(self._add_part(run, (text_of(left), text_of(right))))
        if not alike():
            return self._holders(changed)
        held = Counter(atom for parts in self.words.values() for atom in parts)
        candidates = sorted((atom for atom in set(self.words[left]) ^ set(self.words[right])
                             if atom[1] in self.added),
                            key=lambda atom: (held[atom], -len(atom[1]), atom[1]))
        for atom in candidates:
            for bits in range(self.bits.get(atom, self.ones)+1, self.pair_dim+1):
                if not torch.equal(seen(left, override=(atom, bits)),
                                   seen(right, override=(atom, bits))):
                    self._atom(atom, bits)
                    changed.append(atom)
                    self.revision += 1
                    return self._holders(changed)
        return None

    def _settle(self, pending):
        """Key each pending form, telling apart any two with one code (rule
        56), and write the row of every form whose code changed. Returns
        those forms. No stale key or trie entry is left."""
        affected, pending = set(pending), list(pending)
        while pending:
            word = pending.pop()
            self._unkey(word)
            code = self.code(word)
            key = self.key(code)
            other = self.forms.get(key)
            read = False
            if other is None or other == word:
                # Two codes that differ only where a class that holds both
                # forms cuts the wholes half are one code to their percepts.
                other, read = self._twin(word, code), True
            if other is not None and other != word:
                touched = self._separate(other, word, read=read)
                if touched is None:
                    raise RuntimeError(f'no part tells {text_of(other)!r} from {text_of(word)!r}')
                self._unkey(other)
                # The work stack keeps admission order, so hash order never
                # decides which of two forms a later collision changes.
                touched = set(touched) | {word, other}
                again = [w for w in self.words if w in touched]
                affected.update(again)
                pending.extend(again)
            else:
                self._keyed(word, key, self.key(code[:self.width]))
        for word in sorted(affected):
            if word not in self.word_rows:
                row = self.store.insert(self.WORD_PREFIX + self.word_keys[word],
                                        init_vector=self._storage_code(self.form(word)))
                self.word_rows[word] = row
                self.store.inverse_table[row] = word
                self._fixed_bytes[row] = word
                # Keep byte rows distinct for one-letter words/descent.
                if len(word) > 1:
                    self.store.hash_map[word] = row
                    self.store.radix_trie.insert(word, row)
        if affected:
            ordered = sorted(affected, key=self.order.__getitem__)
            self._packed[torch.tensor([self.order[word] for word in ordered], device='cpu')] = self._pack(
                torch.stack([self.code(word) for word in ordered]))
            self._rewrite(ordered)
        return affected

    # Rule 61: no code inside another by accident -------------------------

    def accidents_of(self, raw):
        """The accidents of a stored form (rule 61), in one pass over every
        stored code, both halves: the stored forms whose code lies inside its
        code and that it does not hold as a run, then those whose code its
        code lies inside and that do not hold it as a run; pairs ``(inner,
        outer)``."""
        return self._accidents_of_many([bytes(raw)])[bytes(raw)]

    def _accidents_of_many(self, forms):
        """:meth:`accidents_of` for several stored forms, against every
        stored code in one pass."""
        live = self._packed[:len(self._ordered)]
        codes = live[torch.tensor([self.order[raw] for raw in forms], device='cpu')]
        inner = ((live[None] & ~codes[:, None]) == 0).all(-1)
        outer = ((codes[:, None] & ~live[None]) == 0).all(-1)
        found = {raw: [] for raw in forms}
        word = self._ordered.__getitem__
        for i, j in inner.nonzero().tolist():
            raw, other = forms[i], word(j)
            if other != raw and other not in self.inside[raw]:
                found[raw].append((other, raw))
        for i, j in outer.nonzero().tolist():
            raw, other = forms[i], word(j)
            if other != raw and other not in self.holders[raw]:
                found[raw].append((raw, other))
        return found

    def accidents(self, chunk=64):
        """Over all ordered pairs of stored forms: the accidents, pairs
        ``(inner, outer)`` whose codes lie one inside the other with no
        link; and the linked parts whose code does not lie inside their
        holder's, pairs ``(part, holder)``."""
        live = self._packed[:len(self._ordered)]
        found, outside = [], []
        for start in range(0, len(live), chunk):
            block = live[start:start + chunk]
            inner = ((live[None] & ~block[:, None]) == 0).all(-1)
            for i, j in inner.nonzero().tolist():
                outer, other = self._ordered[start + i], self._ordered[j]
                if other != outer and other not in self.inside[outer]:
                    found.append((other, outer))
            for i in range(len(block)):
                outer = self._ordered[start + i]
                outside.extend((part, outer) for part in self.inside[outer]
                               if not bool(inner[i, self.order[part]]))
        return found, outside

    def _repair(self, start):
        """Remove every accident from the forms of ``start`` on (rule 61): a
        form whose code changes is tested again, the latest admitted first,
        until none is left. Returns the forms whose code changed."""
        changed = set()
        queue = sorted(start, key=self.order.__getitem__)
        queued = set(queue)
        # The next forms of the queue are tested together; a repair, which
        # changes codes, sets the tests made before it aside.
        tested = {}
        while queue:
            word = queue.pop()
            queued.discard(word)
            if word not in tested:
                tested = self._accidents_of_many(queue[-self.TESTED_AT_ONCE + 1:] + [word])
            by_inner = {}
            for inner, outer in tested.pop(word):
                by_inner.setdefault(inner, []).append(outer)
            for inner, outers in by_inner.items():
                touched = self._separate_inner(inner, outers)
                if touched is None:
                    continue
                tested = {}
                changed.update(touched)
                fresh = [w for w in touched if w not in queued]
                queue.extend(fresh)
                queued.update(fresh)
        return changed

    def _separate_inner(self, inner, outers):
        """Give ``inner`` coordinates that each of ``outers`` lacks (rule
        61), chosen and not drawn, among the zeros of either half: one at a
        time, the coordinate that the most outers still lack (among those its
        basic percept keeps, :meth:`_kept`, where any will do), then one that
        is not on inner's against lane (what its nearest holders add), then
        the one that adds the fewest ones to inner's holders, then the least
        used, then the first. ``inner`` becomes a part of the catalog (a part
        already, it takes the coordinates), and every stored form that holds
        it as a run carries them, so a part stays inside its wholes. Returns
        the forms whose code changed, in order of admission; ``None``, and
        nothing changed, where an outer code has no zero to give."""
        order = self.order.__getitem__
        own = self._packed[order(inner)]
        lacking = self._unpack(~self._packed[torch.tensor([order(o) for o in outers], device='cpu')] & ~own).bool()
        holders = sorted(self.holders[inner], key=order)
        added = (len(holders) - self._unpack(self._packed[torch.tensor([order(h) for h in holders], device='cpu')]).sum(0)
                 if holders else torch.zeros(self.total, dtype=torch.long, device='cpu'))
        # What tells against inner, the parts its nearest holders add (rule
        # 51), is kept: a coordinate of it is taken only where no other will do.
        opposed = torch.zeros(self.total, dtype=torch.bool, device='cpu')
        against = self.against_atoms(inner)
        if against:
            opposed[:self.width] = torch.stack([self.atom_code(atom) for atom in against]).amax(0).eq(1)
        kept = self._kept(inner)
        given = torch.zeros(self.total, dtype=torch.bool, device='cpu')
        pending = torch.ones(len(outers), dtype=torch.bool, device='cpu')
        while bool(pending.any()):
            open_ = lacking[pending] & ~given
            covers = (open_ & kept).sum(0)
            if not bool(covers.any()):
                covers = open_.sum(0)
            if not bool(covers.any()):
                self.failures.append((inner, tuple(o for o, p in zip(outers, pending.tolist()) if p)))
                return None
            choice = covers == covers.max()
            if bool((choice & ~opposed).any()):
                choice &= ~opposed
            choice &= added == added[choice].min()
            choice &= self.usage == self.usage[choice].min()
            best = int(choice.nonzero()[0])
            given[best] = True
            pending &= ~lacking[:, best]
        text = text_of(inner)
        atom = ('part', text)
        touched = sorted(self.holders[inner] | {inner}, key=order)
        value = given.to(torch.float32)
        if atom in self.atoms:
            old = self.given.get(atom)
            self.given[atom] = value if old is None else torch.maximum(old, value)
            self._codes.pop(atom, None)
            self._atom(atom)
        else:
            self.added[text] = (text, text_of(outers[0]))
            self.bits[atom] = 0
            self.given[atom] = value
            self._atom(atom)
            for word in touched:
                self.words[word] = self.words[word] + (atom,)
        index = torch.tensor([order(word) for word in touched], device='cpu')
        old = self._packed[index]
        new = old | self._pack(value)[0]
        self._packed[index] = new
        self._altered.update(touched[i] for i in (old != new).any(1).nonzero().flatten().tolist())
        self.usage += given.long()
        self._given_revision += 1
        return touched

    def _kept(self, raw):
        """Where a coordinate given to a form stays in its basic percept's
        code: all of the parts half, and of the wholes half the AND of the
        codes of the rows of the table of wholes that hold on every byte of
        the form, since a percept's wholes half is at or below each of its
        wholes' (rules 5 and 52). All of the code where no table is bound."""
        kept = torch.ones(self.total, dtype=torch.bool, device='cpu')
        properties = getattr(getattr(self.basic, 'whole', None), 'primitive_properties', None)
        if properties is None or not self.property_width:
            return kept
        stamp = (id(properties), properties.revision)
        if self._table_stamp != stamp:
            with torch.no_grad():
                self._table = (properties.coefficients().detach().cpu() > .5,
                               properties.codes().detach().cpu() >= 1 - 1e-6)
            self._table_stamp = stamp
        holds, codes = self._table
        if codes.shape[-1] != self.property_width:
            return kept
        rows = holds[:, sorted(set(bytes(raw)))].all(1)
        if bool(rows.any()):
            kept[self.width:] = codes[rows].all(0)
        return kept

    def _keep(self, affected, new=None):
        """After a change (rule 61): repair until no accident is left,
        telling apart by rule 56 any two forms the repair leaves with one
        code; write the rows and keys of every form whose code changed; and
        the basic percepts follow (``BasicPercepts.follow_forms``). Returns
        those forms. ``last_changed`` holds those admitted before ``new``;
        ``last_touched`` also the holders that took a coordinate they had."""
        changed, pending, touched = set(affected), set(affected), set()
        while pending:
            self._altered = set()
            touched |= self._repair(pending)
            changed |= self._altered
            collided = self._rewrite(self._altered)
            pending = self._settle({word for pair in collided for word in pair}) if collided else set()
            changed |= pending
        self.last_touched = frozenset((touched | set(affected)) - {new})
        self.last_changed = frozenset(changed - {new})
        if self.last_changed:
            self.revision += 1
        self.basic.follow_forms(changed, new)
        return changed

    def admit(self, raw):
        raw = bytes(raw)
        if not raw:
            raise ValueError('a word identity requires nonempty bytes')
        if raw not in self.words:
            # A stored form of several characters is a part of its own,
            # carried by itself and by every stored form that holds it as a
            # run, those admitted before it and those admitted after.
            text = text_of(raw)
            own = ('part', text)
            if len(text) > 1 and text not in self.added:
                self.added[text] = None
            self.words[raw] = self.atoms_of(raw)
            self._link(raw)
            carriers = [holder for holder in sorted(self.holders[raw], key=self.order.__getitem__)
                        if own not in self.words[holder]] if len(text) > 1 else []
            for holder in carriers:
                self.words[holder] = self.words[holder] + (own,)
            for atom in self.words[raw]:
                self._atom(atom)
            self._place(raw)
            if carriers:
                index = torch.tensor([self.order[word] for word in carriers], device='cpu')
                self._packed[index] = self._pack(torch.stack([self.code(word) for word in carriers]))
            self._keep(self._settle(carriers + [raw]), new=raw)
        return self.parts(raw)

    def admit_part(self, run):
        """Add a run of characters as a part. Every form held that contains
        it takes it and is rekeyed. A part that exists, or a single
        character, changes nothing. Returns the forms that took it."""
        run = text_of(run) if isinstance(run, (bytes, bytearray)) else str(run)
        if len(run) < 2 or run in self.added:
            return ()
        atom = self._add_part(run, None)
        holders = self._holders([atom])
        self._keep(self._settle(holders))
        self.revision += 1
        return tuple(holders)

    def part_between(self, string, word):
        """The part that tells a string from a word read in its place: the
        shortest run the string holds and the word does not."""
        return distinguishing_run(string, word, one_sided=True)

    def parts(self, raw):
        """The form's part rows and its witnessed nested wholes."""
        return (tuple(self.atoms[a] for a in self.words[raw]) +
                tuple(self.word_rows[part] for part, _lo, _hi in self.nested.get(raw, ())))

    def part_atoms(self, raw):
        atoms = set(self.words[raw])
        for part, _lo, _hi in self.nested.get(raw, ()):
            atoms.update(self.part_atoms(part))
        return atoms

    @contextmanager
    def nested_lesson(self, marker):
        """Permit admission of one suffix pattern for this lesson only.

        The permission is not a learned identity or checkpoint setting.
        Already witnessed parts persist when the lesson ends.
        """
        if not isinstance(marker, (str, bytes)) or not marker:
            raise ValueError('a nested lesson names one nonempty marker')
        if self._marker_pattern is not None:
            raise ValueError('only one nested marker lesson may be active')
        self._marker_pattern = bytes_of(marker) if isinstance(marker, str) else marker
        try:
            yield self
        finally:
            self._marker_pattern = None

    def observe_nested(self, raw):
        """Witness a form; record the lesson's marker after a known stem.

        The record is read as the form's nested wholes (``parts``); a form's
        code already holds its stem's and its marker's, so nothing is joined.
        """
        raw = bytes(raw)
        self.admit(raw)
        if not text_of(raw).replace(' ', '').isalpha():
            return ()
        self.word_witnesses[raw] = self.word_witnesses.get(raw, 0) + 1
        marker = self._marker_pattern
        if marker is None:
            return tuple((self.word_rows[part],lo,hi) for part,lo,hi in self.nested.get(raw,()))
        nested = dict(self.nested)
        # A new stem can license an earlier witnessed suffixed word. There
        # is no vocabulary-wide substring search or token-count threshold.
        for word in (raw, raw + marker):
            if word not in self.word_witnesses or not word.endswith(marker):
                continue
            stem = word[:-len(marker)]
            if stem and stem in self.word_witnesses and word not in nested:
                nested[word] = ((stem, 0, len(stem)), (marker, len(stem), len(word)))
        if nested != self.nested:
            # Admit the named marker through the same native identity owner.
            for parts in nested.values():
                for part, _lo, _hi in parts:
                    self.admit(part)
            self.nested = nested
            self.revision += 1
        return tuple((self.word_rows[part],lo,hi) for part,lo,hi in self.nested.get(raw,()))

    def row_for_form(self, form):
        """The row of the stored form whose code (:meth:`code`) this is. A
        parts half alone names a form only where no other stored form has
        that parts half."""
        if self.property_width and form.shape[-1] >= self.width + self.property_width:
            raw = self.forms.get(self.key(form[..., :self.width + self.property_width]))
        else:
            held = self.by_parts.get(self.key(form[..., :self.width]), ())
            raw = next(iter(held)) if len(held) == 1 else None
        return None if raw is None else self.word_rows[raw]

    def read(self, form):
        row = self.row_for_form(form)
        return None if row is None else self.store.bytes_for(row)

    def spelling(self, rows):
        """The stored form whose code is read from these rows: the fold of
        their codes and their tightest whole.

        A code that is no stored form has no spelling: ``None``.
        """
        rows = list(map(int, rows))
        form = self.store._basis.fixed_row_codes[rows].amax(0).detach().cpu()[:self.width]
        if self.property_width:
            wholes = self.tightest(rows)
            if wholes is None:
                return None
            form = torch.cat((form, wholes))
        return self.read(form)

    def vocab_extras(self):
        return dict(atoms=self.atoms, bits=self.bits, words=self.words,
                    word_rows=self.word_rows, added=self.added,
                    revision=self.revision,
                    word_witnesses=self.word_witnesses, nested=self.nested,
                    holders={word: tuple(sorted(found)) for word, found in self.holders.items()},
                    inside={word: tuple(sorted(found)) for word, found in self.inside.items()},
                    given={atom: tuple(value.nonzero().flatten().tolist()) for atom, value in self.given.items()},
                    usage=self.usage.tolist())

    def _relink(self):
        """The links of every stored form, from the forms alone, for a
        vocabulary saved without them."""
        self.holders = {word: set() for word in self.words}
        self.inside = {word: set() for word in self.words}
        for word in self.words:
            for size in range(1, len(word)):
                for start in range(len(word) - size + 1):
                    run = word[start:start + size]
                    if run in self.holders:
                        self.inside[word].add(run)
                        self.holders[run].add(word)

    def load_vocab_extras(self, value):
        for name in ('atoms', 'bits', 'words', 'word_rows', 'added', 'revision'):
            setattr(self, name, value[name])
        self.__dict__.pop('_by_byte', None)
        self._codes = {}
        self.word_witnesses = value.get('word_witnesses', {})
        self.nested = value.get('nested', {})
        self._marker_pattern = None
        self._fixed_bytes = {row: bytes_of(atom[1]) for atom, row in self.atoms.items()}
        self._fixed_bytes.update({row: raw for raw, row in self.word_rows.items()})
        self._property_revision = None
        self._atom_rows, self._catalog = {}, {}
        if 'holders' in value and 'inside' in value:
            self.holders = {word: set(found) for word, found in value['holders'].items()}
            self.inside = {word: set(found) for word, found in value['inside'].items()}
        else:
            self._relink()
        self.given = {}
        for atom, coordinates in value.get('given', {}).items():
            self.given[atom] = torch.zeros(self.total, device='cpu')
            self.given[atom][list(coordinates)] = 1
        self.usage = torch.tensor(value.get('usage', [0] * self.total), dtype=torch.long, device='cpu')
        self._given_revision += 1
        self.failures = []
        self.last_changed = self.last_touched = frozenset()
        self._reindex()

    def audit(self):
        words = sorted(self.words)
        forms = {w: self.form(w) for w in words}
        postings = {}
        for word in words:
            for atom in self.words[word]:
                postings.setdefault(atom, set()).add(word)
        comparable = []
        for word in words:
            lists = sorted((postings[a] for a in self.words[word]), key=len)
            supersets = lists[0].intersection(*lists[1:])
            comparable.extend((word, other) for other in sorted(supersets) if other != word)
        return dict(vocabulary=[text_of(w) for w in words], pair_dim=self.pair_dim,
            ones=self.ones,
            collisions=len(words)-len({self.key(v) for v in forms.values()}),
            containment_pairs=[(text_of(a), text_of(b)) for a, b in comparable],
            containment_violations=sum(bool((forms[a] > forms[b]).any()) for a,b in comparable),
            reconstruction_errors=sum(self.read(forms[w]) != w for w in words), added=self.added)
