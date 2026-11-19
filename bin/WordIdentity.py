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
        # Each stored form's holders: the stored forms that hold it as a run.
        self.holders = {}
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
        result = torch.zeros(self.width)
        result[list(bit_stream(bytes_of(value), self.pair_dim)[:n])] = 1
        if bits is None:
            self._codes[atom] = result
        return result

    def catalog_code(self, atom, bits=None):
        """A part's code in the wholes half: the catalog drawn a second time
        in that half's own coordinates. A character's is its byte's in the
        table of wholes; a part given more bits in the parts half is given
        as many more here (``bits``: read at those bits)."""
        bits = self.bits.get(atom, self.ones) if bits is None else int(bits)
        n = self.property_ones + bits - self.ones
        key = (atom, n)
        if key not in self._catalog:
            from PerceptProperties import catalog_coordinates
            result = torch.zeros(self.property_width)
            result[list(catalog_coordinates(bytes_of(atom[1]), self.property_width, n))] = 1
            self._catalog[key] = result
        return self._catalog[key]

    def tightest(self, rows):
        """The tightest whole of the form whose parts are these store rows:
        the OR of their catalog codes in the wholes half. ``None`` where a
        row is no part or form of this catalog."""
        if not self.property_width:
            return None
        key = (tuple(map(int, rows)), self.revision, len(self.added), len(self.atoms), len(self.word_rows))
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

    def _hold(self, raw):
        """Record a new stored form among the holders: of each stored form
        it holds as a run, and of itself those stored forms that hold it."""
        text = bytes(raw)
        self.holders[text] = {word for word in self.words if word != text and text in word}
        for size in range(1, len(text)):
            for start in range(len(text) - size + 1):
                run = text[start:start + size]
                if run in self.holders:
                    self.holders[run].add(text)

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
        return tuple(sorted({atom for holder in self.nearest_holders(raw)
                             for atom in self.words[holder] if atom not in own}))

    def atoms_of(self, raw):
        """The parts of a form: its characters and the added parts inside it."""
        text = text_of(raw)
        return base_atoms(raw) + tuple(('part', run) for run in self.added if run in text)

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

    def _reindex(self):
        self.forms = {}
        self.word_keys = {}
        self.by_parts = {}
        for raw in self.words:
            key = self.key(self.code(raw))
            self.word_keys[raw] = key
            self.by_parts.setdefault(self.key(self.form(raw)), set()).add(raw)
            if key in self.forms and self.forms[key] != raw:
                return self.forms[key], raw
            self.forms[key] = raw
        return None

    def _unkey(self, word):
        old = self.word_keys.pop(word, None)
        if old is not None and self.forms.get(old) == word:
            del self.forms[old]
            self.store.hash_map.pop(self.WORD_PREFIX+old, None)
            self.store.radix_trie.remove(self.WORD_PREFIX+old)
        for words in self.by_parts.values():
            words.discard(word)

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

    def _separate(self, left, right):
        """Tell two forms with one code apart (rule 56); return the forms
        whose code changed, or ``None`` where no part tells them apart.

        Rule 43's next part is added, the shortest run that one holds and
        the other does not, until their codes differ. Where no such run is
        left, an added part that one holds and the other does not is given
        more bits, the part held by the fewest forms first.
        """
        changed = []
        alike = lambda: self.key(self.code(left)) == self.key(self.code(right))
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
                if not torch.equal(self.code(left, override=(atom, bits)),
                                   self.code(right, override=(atom, bits))):
                    self._atom(atom, bits)
                    changed.append(atom)
                    self.revision += 1
                    return self._holders(changed)
        return None

    def _settle(self, pending):
        """Key each pending form, telling apart any two with one row, and
        write the row of every form whose code changed. No stale key or trie
        entry is left."""
        affected, pending = set(pending), list(pending)
        while pending:
            word = pending.pop()
            self._unkey(word)
            key = self.key(self.code(word))
            other = self.forms.get(key)
            if other is not None and other != word:
                touched = self._separate(other, word)
                if touched is None:
                    raise RuntimeError(f'no part tells {text_of(other)!r} from {text_of(word)!r}')
                self._unkey(other)
                # The work stack keeps admission order, so hash order never
                # decides which of two forms a later collision changes.
                again = [w for w in self.words if w in set(touched) | {word, other}]
                affected.update(again)
                pending.extend(again)
            else:
                self.forms[key] = word
                self.word_keys[word] = key
                self.by_parts.setdefault(self.key(self.form(word)), set()).add(word)
        for word in sorted(affected):
            value = self.form(word)
            code = self.key(self.code(word))
            key = self.WORD_PREFIX + code
            row = self.word_rows.get(word)
            if row is None:
                row = self.store.insert(key, init_vector=self._storage_code(value))
                self.word_rows[word] = row
                self.store.inverse_table[row] = word
                # Keep byte rows distinct for one-letter words/descent.
                if len(word) > 1:
                    self.store.hash_map[word] = row
                    self.store.radix_trie.insert(word, row)
            self._fixed(row, value, word)
            self.store.hash_map[key] = row
            self.store.radix_trie.insert(key, row)

    def admit(self, raw):
        raw = bytes(raw)
        if not raw:
            raise ValueError('a word identity requires nonempty bytes')
        if raw not in self.words:
            self.words[raw] = self.atoms_of(raw)
            self._hold(raw)
            for atom in self.words[raw]:
                self._atom(atom)
            self._settle([raw])
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
        self._settle(holders)
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
                    word_witnesses=self.word_witnesses, nested=self.nested)

    def load_vocab_extras(self, value):
        for name in ('atoms', 'bits', 'words', 'word_rows', 'added', 'revision'):
            setattr(self, name, value[name])
        self._codes = {}
        self.word_witnesses = value.get('word_witnesses', {})
        self.nested = value.get('nested', {})
        self._marker_pattern = None
        self._fixed_bytes = {row: bytes_of(atom[1]) for atom, row in self.atoms.items()}
        self._fixed_bytes.update({row: raw for raw, row in self.word_rows.items()})
        self._property_revision = None
        self._atom_rows, self._catalog = {}, {}
        self.holders = {word: set() for word in self.words}
        for word in self.words:
            for size in range(1, len(word)):
                for start in range(len(word) - size + 1):
                    run = word[start:start + size]
                    if run in self.holders:
                        self.holders[run].add(word)
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
