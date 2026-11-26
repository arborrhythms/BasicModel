"""Store rows and ordered parts of lexical forms.

A stored form is known by its row. In a configured text model its parts
coordinates are read live from FormLinks, including subsequent pull steps;
the code is not its identity. Atom and word rows belong to the existing
RadixLayer. Standalone stores without a form table retain the legacy atom
codes for their independent callers.
"""
import hashlib
from functools import lru_cache
from itertools import groupby

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
    if pair < ones or ones < 1:
        raise ValueError('identity requires D >= s > 0')
    if property_width < 1:
        raise ValueError('property codes require propertyDim > 0')
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

    def __init__(self, store, pair_dim=64, ones=3, property_width=0, basic_capacity=None):
        super().__init__()
        object.__setattr__(self, 'store', store)
        self.pair_dim, self.ones = int(pair_dim), int(ones)
        self.width = self.pair_dim
        self.property_width = int(property_width)
        if (store.dim != self.width + self.property_width or self.property_width < 0
                or not 0 < self.ones <= self.pair_dim):
            raise ValueError('Radix bank width must be identity + property, with 0 < s <= D')
        object.__setattr__(self, 'properties', None)
        self._fixed_bytes = {}
        self._codes = {}
        # A stored form is known by its bytes: one row of the store for each
        # spelling, as the table of forms has one row for each. Its code is
        # read from its parts and names nothing; two forms may have one code.
        self.atoms, self.words, self.word_rows = {}, {}, {}
        # Each stored form's links, written when it is admitted and saved:
        # the stored forms that hold it as a run (its holders, its wholes
        # among the forms) and those it holds as a run (its parts among the
        # forms). A form that holds another's characters and not the run is
        # neither.
        self.holders = {}
        self.inside = {}
        # Each stored form's place in the order of admission.
        self.order = {}
        self._atom_rows = {}
        self._catalog = {}
        # Each added part, in the order added: a stored form of several
        # characters, or a taught run.
        self.added = {}
        self.nested = {}
        self.revision = 0
        from BasicPercepts import BasicPercepts
        self.basic = BasicPercepts(store, self.width, self.property_width,
            store._capacity if basic_capacity is None else basic_capacity)
        basis = store._basis
        if not hasattr(basis, 'fixed_row_mask'):
            basis.register_buffer('fixed_row_mask', torch.zeros(store._capacity, dtype=torch.bool, device=basis.W.device))
            basis.register_buffer('fixed_row_codes', torch.zeros_like(basis.W, requires_grad=False))

    def atom_code(self, atom):
        """A part's live table point; isolated stores without a table keep their legacy codes."""
        if getattr(self, 'forms_table', None) is not None:
            return self._table_code(bytes_of(atom[1]))
        if atom not in self._codes:
            result = torch.zeros(self.width, device='cpu')
            result[list(bit_stream(bytes_of(atom[1]), self.pair_dim)[:self.ones])] = 1
            self._codes[atom] = result
        return self._codes[atom]

    def bind_forms(self, table):
        """Read form coordinates live from the table owned by PartSpace."""
        object.__setattr__(self, 'forms_table', table)
        self._table_rows = {}
        self._row_spellings = None
        self.__dict__.pop('_registered_roots', None)
        basis = self.store._basis
        object.__setattr__(basis, 'form_links', table)
        basis.fixed_parts_width = self.width
        if not hasattr(basis, 'fixed_form_rows'):
            basis.register_buffer('fixed_form_rows', torch.full((self.store._capacity,), -1,
                                  dtype=torch.long, device=basis.W.device))
        for row, raw in self._fixed_bytes.items():
            basis.fixed_form_rows[row] = self._table_row(raw)

    def _table_row(self, raw):
        raw = bytes(raw)
        if len(raw) == 1:
            return raw[0]  # FormLinks seats every byte at its integer value.
        if raw not in self._table_rows:
            table = self.forms_table
            data = table.left.new_tensor([list(raw)])
            lengths = table.left.new_tensor([len(raw)])
            row = int(table.read(data, lengths, admit=False)[0])
            if row < 0:
                row = int(table.insert(data, lengths)[0])
            self._remember_form(raw, row)
        return self._table_rows[raw]

    def _remember_form(self, raw, row):
        if self._table_rows.setdefault(raw, row) != row:
            raise ValueError('nomination disagrees with its stored form row')
        if self._row_spellings is not None:
            self._row_spellings[row] = raw
            self._unseated.discard(raw)

    def _registration_index(self):
        """Restore the form-address index once, including pre-table spellings."""
        if self._row_spellings is None:
            self._row_spellings = {byte: bytes([byte]) for byte in range(256)}
            self._row_spellings.update((row, spelling) for spelling, row in self._table_rows.items())
            # Old vocabularies can have forms which predate the form table.
            # Read their row when present and retain the old membership for
            # any missing form. This lookup never inserts or counts a reading.
            known = set(self.words) | {bytes_of(run) for run in self.added}
            missing = [word for word in known if len(word) > 1 and word not in self._table_rows]
            self._unseated = set()
            if missing:
                table = self.forms_table
                width = max(map(len, missing))
                batch_size = max(1, (1 << 20) // width)
                for start in range(0, len(missing), batch_size):
                    group = missing[start:start + batch_size]
                    data = table.left.new_tensor([list(word.ljust(width, b'\0')) for word in group])
                    rows = table.read(data, table.left.new_tensor(list(map(len, group))), admit=False)
                    for word, row in zip(group, rows.tolist(), strict=True):
                        if row >= 0:
                            self._remember_form(word, row)
                        else:
                            self._unseated.add(word)

    def registered_roots(self):
        """Sorted device form addresses and the longest seated spelling."""
        self._registration_index()
        table = self.forms_table
        cached = self.__dict__.get('_registered_roots')
        if (cached is None or cached[0] is not self.words or cached[1] != len(self.words)
                or cached[2] != len(self._unseated) or cached[3].device != table.left.device):
            seated = [raw for raw in self.words if raw not in self._unseated]
            roots = table.left.new_tensor(sorted(raw[0] if len(raw) == 1 else self._table_rows[raw]
                                                 for raw in seated), dtype=torch.long)
            cached = self.words, len(self.words), len(self._unseated), roots, max(map(len, seated), default=0)
            self.__dict__['_registered_roots'] = cached
        return cached[3], cached[4]

    def _registered_parts(self, raw, part_rows):
        """Resolve device containment against the inventory at this nomination."""
        self._registration_index()
        inside, runs = set(), set()
        for address in set(part_rows):
            if address < 0:
                continue
            spelling = self._row_spellings.get(address // 2)
            if spelling is None:
                continue
            if spelling != raw and spelling in self.inside:
                inside.add(spelling)
            if address % 2:
                run = text_of(spelling)
                if run in self.added:
                    runs.add(run)
        if self._unseated:
            text = text_of(raw)
            for spelling in self._unseated:
                if spelling != raw and spelling in self.inside and spelling in raw:
                    inside.add(spelling)
                run = text_of(spelling)
                if run in self.added and run in text:
                    runs.add(run)
        rank = self._added_rank()
        return inside, tuple(('part', run) for run in sorted(runs, key=rank.__getitem__))

    def _table_code(self, raw):
        return self._table_codes(self._table_row(raw))

    def _table_codes(self, rows):
        table = self.forms_table
        half = min(self.width, table.code.shape[-1] // 2)
        code = table.code[rows, :half]
        return torch.nn.functional.pad(code, (0, self.width - half))

    def _wholes_table(self):
        """The table of wholes the catalog reads, or ``None`` where none is bound."""
        properties = self.properties
        if properties is None:
            properties = getattr(getattr(self.basic, 'whole', None), 'primitive_properties', None)
        return properties if properties is not None and properties.property_width == self.property_width else None

    def _layout(self):
        """The table's count of byte regroupings; -1 with no table."""
        properties = self._wholes_table()
        return -1 if properties is None else properties.layout

    def catalog_code(self, atom):
        """A part's code in the wholes half: the OR of its bytes' codes in the
        table of wholes, a byte one at its group's coordinate (every byte at
        the first where no table is bound)."""
        properties = self._wholes_table()
        result = torch.zeros(self.property_width, device='cpu')
        raw = bytes_of(atom[1])
        result[[0] * len(raw) if properties is None else [properties.byte_coordinate[b] for b in raw]] = 1
        return result

    def _row_atoms(self, rows):
        """The parts these store rows stand for, a part's row for itself and
        a stored form's row for its parts. ``None`` where a row is no part or
        form of this catalog."""
        if len(self._atom_rows) != len(self.atoms) + len(self.word_rows):
            self._atom_rows = {row: (atom,) for atom, row in self.atoms.items()}
            self._atom_rows.update({row: raw for raw, row in self.word_rows.items()})
        atoms = set()
        for row in map(int, rows):
            held = self._atom_rows.get(row)
            if held is None:
                return None
            atoms.update(held if isinstance(held, tuple) else self.part_atoms(held))
        return atoms

    def tightest(self, rows):
        """The tightest whole of the form whose parts are these store rows:
        the OR of their catalog codes in the wholes half. ``None`` where a
        row is no part or form of this catalog."""
        if not self.property_width:
            return None
        key = (tuple(map(int, rows)), self.revision, len(self.added), len(self.atoms), len(self.word_rows),
               self._layout())
        if key in self._catalog:
            return self._catalog[key]
        atoms = self._row_atoms(rows)
        if atoms is None:
            return None
        result = torch.stack([self.catalog_code(atom) for atom in atoms]).amax(0) if atoms else None
        self._catalog[key] = result
        return result

    def escaped_elements(self, device):
        return self._occurrences(device, False) is None

    def occurrence_bank(self, device, *, escaped_elements=None):
        if escaped_elements is None:
            escaped_elements = self.escaped_elements(device)
        bank = self._occurrences(device, escaped_elements)
        return None if bank is None else bank[:4]

    def _occurrences(self, device, escaped_elements):
        """Group native part references by spelling length for the occurrence read.

        The cache holds addresses and fixed catalog coordinates. Native part
        codes are read through the actual basis on each differentiable forward.
        """
        stamp = (len(self.atoms), len(self.added), len(self.word_rows), self.revision, self._layout(), device)
        key = ('occurrence_bank', escaped_elements)
        cached = self._catalog.get(key)
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
                if not escaped_elements:
                    self._catalog[key] = stamp, None
                    return None
                raw = bytes_of(text)
            groups.setdefault(len(raw), []).append((raw, row, atom))
        spellings, rows, catalogs, ranks, forms = [], [], [], [], []
        added_rank = self._added_rank()
        for size in sorted(groups):
            entries = groups[size]
            spellings.append(torch.tensor([list(raw) for raw, _, _ in entries],
                                          dtype=torch.uint8, device=device))
            rows.extend(row for _, row, _ in entries)
            forms.extend(self.word_rows.get(raw, -1) for raw, _, _ in entries)
            ranks.extend(-1 if len(atom[1]) == 1 else added_rank[atom[1]] for _, _, atom in entries)
            if self.property_width:
                catalogs.extend(self.catalog_code(atom) for _, _, atom in entries)
        addresses = torch.tensor(rows, dtype=torch.long, device=device)
        catalog = (torch.stack(catalogs).to(device) if catalogs else
                   torch.empty(len(rows), self.property_width, device=device))
        bank = tuple(spellings), addresses, catalog, addresses.new_tensor(ranks), addresses.new_tensor(forms)
        self._catalog[key] = stamp, bank
        return bank

    def ordered_cover(self, raw):
        """Proper constituents and byte intervals at the host lesson/checkpoint boundary."""
        nested = self.nested.get(raw, ())
        if (nested and nested[0][1] == 0 and nested[-1][2] == len(raw)
                and all(left[2] == right[1] for left, right in zip(nested, nested[1:]))
                and all(0 <= lo < hi <= len(raw) and hi - lo == len(part) and part != raw
                        for part, lo, hi in nested)):
            return [(part, start, end) for part, start, end in nested]
        text = text_of(raw)
        candidates = {atom[1] for atom in self.atoms_of(raw) if 1 < len(atom[1]) < len(text)}
        result = []
        at = start = 0
        while at < len(text):
            part = max((part for part in candidates if text.startswith(part, at)),
                       key=len, default=text[at])
            end = start + len(bytes_of(part))
            result.append((('part', part), start, end))
            at += len(part)
            start = end
        return result

    def retain_spelling(self, raw):
        """Keep ordered native references on a form and on its non-element parts."""
        groups = self.store.part_groups
        row = self.word_rows[raw]
        text = text_of(raw)
        if len(text) < 2:
            return
        if self.nested.get(raw):
            cover = self.ordered_cover(raw)
            if not isinstance(cover[0][0], bytes):
                # Older overlapping definitions retain their original inverse.
                return
            groups[row] = tuple(self.word_rows[part] for part, _, _ in cover)
            self.store._canonical_group_cache = None
            return
        children = tuple(self.atoms[part] for part, _, _ in self.ordered_cover(raw))
        for child in children:
            if child not in groups:
                part = text_of(self._fixed_bytes[child])
                if len(part) > 1:
                    groups[child] = tuple(self.atoms['part', char] for char in part)
        groups[row] = children
        self.store._canonical_group_cache = None

    def _link(self, raw, *, inside=None, holders=None):
        """Link a new stored form to the stored forms it holds as a run (its
        parts among the forms) and to those that hold it (its holders), on
        both sides of each link."""
        text = bytes(raw)
        if inside is None:
            inside = {text[start:start + size] for size in range(1, len(text))
                      for start in range(len(text) - size + 1)
                      if text[start:start + size] in self.inside}
        if holders is None:
            by_byte = self._forms_by_byte(text)
            lists = [self.holders[run] for run in inside]
            lists.extend(by_byte[byte] for byte in set(text))
            shortest = min(lists, key=len)
            holders = {word for word in shortest if word != text and text in word}
        self.inside[text], self.holders[text] = inside, holders
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
            if hasattr(basis, 'fixed_form_rows'):
                basis.fixed_form_rows[row] = self._table_row(raw)
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
        _ = self.properties.revision  # Observe writes at the same boundary as the live property reader.

    def _storage_code(self, identity_code):
        """Admission reserves both subspaces; _fixed writes the property fill."""
        return torch.nn.functional.pad(identity_code, (0, self.property_width))

    def _atom(self, atom):
        code = self.atom_code(atom)
        if atom not in self.atoms:
            key = self.PREFIX + atom[0].encode() + b'/' + bytes_of(atom[1])
            self.atoms[atom] = self.store.insert(key, init_vector=self._storage_code(code))
        self._fixed(self.atoms[atom], code, bytes_of(atom[1]))
        return self.atoms[atom]

    @torch.no_grad()
    def _admit_atoms(self, atoms):
        """Refresh established part rows together, in between new radix admissions."""
        basis = self.store._basis
        for seated, group in groupby(atoms, key=self.atoms.__contains__):
            if not seated:
                for atom in group:
                    self._atom(atom)
                continue
            members = tuple(group)
            rows = [self.atoms[atom] for atom in members]
            if len(members) == 1 or len(set(rows)) != len(rows):
                for atom in members:
                    self._atom(atom)
                continue
            addresses = torch.tensor(rows, device=basis.W.device, dtype=torch.long)
            forms = None
            if getattr(self, 'forms_table', None) is not None:
                forms = addresses.new_tensor([self._table_row(bytes_of(atom[1])) for atom in members])
                codes = self._table_codes(forms).to(basis.W)
            else:
                codes = torch.stack([self.atom_code(atom) for atom in members]).to(basis.W)
            codes = torch.nn.functional.pad(codes, (0, self.property_width))
            basis.fixed_row_mask.index_fill_(0, addresses, True)
            basis.fixed_row_codes.index_copy_(0, addresses, codes)
            basis.W.index_copy_(0, addresses, codes)
            self._fixed_bytes.update((row, bytes_of(atom[1])) for row, atom in zip(rows, members))
            if forms is not None:
                basis.fixed_form_rows.index_copy_(0, addresses, forms)

    def _seat_stored_form(self, raw, carriers):
        """Write the nominated word, its parts and its holders from their chart rows."""
        members = tuple(dict.fromkeys(self.words[raw]))
        spellings = [bytes_of(atom[1]) for atom in members]
        keys = [self.atoms[atom] if atom in self.atoms else self.PREFIX + atom[0].encode() + b'/' + spelling
                for atom, spelling in zip(members, spellings, strict=True)]
        spellings.extend((raw, *carriers))
        keys.append(self.WORD_PREFIX + raw)
        keys.extend(self.word_rows[holder] for holder in carriers)
        forms = self.forms_table.left.new_tensor([self._table_row(word) for word in spellings], dtype=torch.long)
        codes = self._storage_code(self._table_codes(forms)).to(self.store._basis.W)
        rows = self.store.write_fixed_forms(keys, forms, codes)
        self.atoms.update(zip(members, rows[:len(members)], strict=True))
        self._fixed_bytes.update(zip(rows, spellings, strict=True))
        return rows[len(members)]

    def form(self, raw):
        """A form's live table point, or the legacy fold in an isolated store."""
        raw = bytes(raw)
        if getattr(self, 'forms_table', None) is not None:
            return self._table_code(raw)
        atoms = self.words.get(raw) or self.atoms_of(raw)
        return torch.stack([self.atom_code(a) for a in atoms]).amax(0)

    def definition_key(self, raw):
        """A stored form's identity for its word's definition: its row of the store."""
        return self.word_rows[bytes(raw)]

    @torch.no_grad()
    def _write_forms(self, words):
        """Write the store rows of these stored forms from their parts' codes, in one indexed write."""
        words = list(words)
        if not words:
            return
        basis = self.store._basis
        rows = torch.tensor([self.word_rows[word] for word in words], device=basis.W.device)
        forms = None
        if getattr(self, 'forms_table', None) is not None:
            forms = rows.new_tensor([self._table_row(word) for word in words])
            value = self._table_codes(forms).to(basis.W)
        else:
            value = torch.stack([self.form(word) for word in words]).to(basis.W)
        value = torch.nn.functional.pad(value, (0, self.property_width))
        basis.fixed_row_mask[rows] = True
        basis.fixed_row_codes[rows] = value
        basis.W[rows] = value
        if forms is not None:
            basis.fixed_form_rows.index_copy_(0, rows, forms)

    def _record_added(self, run):
        count = len(self.added)
        self.added[run] = None
        if self.__dict__.get('_rank_stamp') == (id(self.added), count):
            self._rank[run] = count
            self._rank_stamp = id(self.added), len(self.added)

    def _add_part(self, run):
        self._record_added(run)
        atom = ('part', run)
        self._atom(atom)
        for word in self.words:
            if run in text_of(word):
                self.words[word] = self.atoms_of(word)
        return atom

    def _holders(self, atoms):
        atoms = set(atoms)
        return [word for word, parts in self.words.items() if atoms.intersection(parts)]

    def admit(self, raw, *, form_row=None, witnesses=None, holder_rows=None):
        raw = bytes(raw)
        if not raw:
            raise ValueError('a word identity requires nonempty bytes')
        if form_row is not None:
            table = self.forms_table
            if not 0 <= form_row < table.capacity:
                raise IndexError('word nomination is outside the table of forms')
            self._remember_form(raw, form_row)
        if raw not in self.words:
            # A stored form of several characters is a part of its own,
            # carried by itself and by every stored form that holds it as a
            # run, those admitted before it and those admitted after.
            text = text_of(raw)
            own = ('part', text)
            if len(text) > 1 and text not in self.added:
                self._record_added(text)
            inside = None
            if witnesses is None:
                self.words[raw] = self.atoms_of(raw)
            else:
                characters, part_rows = witnesses
                base = tuple(('part', text_of(octets)) for octets, _ in characters)
                for octets, row in characters:
                    self._remember_form(octets, row)
                inside, added = self._registered_parts(raw, part_rows)
                self.words[raw] = base + added
            holders = None
            if holder_rows is not None:
                # Other new roots from this batch may become words later.
                # A reading can update only a holder already in the catalog.
                holders = {self._row_spellings[row] for row in holder_rows
                           if self._row_spellings.get(row) in self.inside}
                holders.update(holder for holder in self._unseated if holder in self.inside and raw in holder)
            self._link(raw, inside=inside, holders=holders)
            carriers = [holder for holder in sorted(self.holders[raw], key=self.order.__getitem__)
                        if own not in self.words[holder]] if len(text) > 1 else []
            for holder in carriers:
                self.words[holder] = self.words[holder] + (own,)
            if witnesses is None:
                self._admit_atoms(self.words[raw])
                row = self.store.insert(self.WORD_PREFIX + raw, init_vector=self._storage_code(self.form(raw)))
            else:
                row = self._seat_stored_form(raw, carriers)
            self.order[raw] = len(self.order)
            self.word_rows[raw] = row
            self.store.inverse_table[row] = raw
            self._fixed_bytes[row] = raw
            # Keep byte rows distinct for one-letter words/descent.
            if len(raw) > 1:
                self.store.hash_map[raw] = row
                self.store.radix_trie.insert(raw, row)
            # The forms that hold the new one carry its part: their rows are
            # written again, and the basic percepts follow.
            if witnesses is None:
                self._write_forms(carriers + [raw])
            if carriers:
                self.revision += 1
            self.basic.follow_forms(set(carriers) | {raw}, raw)
            self.retain_spelling(raw)
        return self.parts(raw)

    def admit_part(self, run):
        """Add a run of characters as a part. Every form held that contains
        it takes it and its row is written again. A part that exists, or a
        single character, changes nothing. Returns the forms that took it."""
        run = text_of(run) if isinstance(run, (bytes, bytearray)) else str(run)
        if len(run) < 2 or run in self.added:
            return ()
        atom = self._add_part(run)
        holders = self._holders([atom])
        self._write_forms(holders)
        self.basic.follow_forms(set(holders))
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

    def spelling_row(self, rows):
        """The row of the stored form whose parts these store rows are: the
        form of the longest part among them, where its parts are exactly
        theirs. ``None`` where a row is no part or form of this catalog, or
        the parts are no stored form's."""
        atoms = self._row_atoms(rows)
        if not atoms:
            return None
        raw = bytes_of(max(atoms, key=lambda atom: len(atom[1]))[1])
        return self.word_rows[raw] if raw in self.word_rows and atoms == self.part_atoms(raw) else None

    def spelling(self, rows):
        root = self.spelling_row(rows)
        return None if root is None else self.store.bytes_for(root)

    def vocab_extras(self):
        return dict(atoms=self.atoms, words=self.words, word_rows=self.word_rows, added=self.added,
                    revision=self.revision, nested=self.nested,
                    holders={word: tuple(sorted(found)) for word, found in self.holders.items()},
                    inside={word: tuple(sorted(found)) for word, found in self.inside.items()})

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
        if value.get('given') or value.get('bits'):
            raise ValueError('this vocabulary was saved with coordinates given to its forms to tell their codes '
                             'apart; a stored form is now known by its row, and those codes are not restored')
        for name in ('atoms', 'words', 'word_rows', 'added', 'revision'):
            setattr(self, name, value[name])
        self.__dict__.pop('_by_byte', None)
        self.__dict__.pop('_registered_roots', None)
        if getattr(self, 'forms_table', None) is not None:
            self._row_spellings = None
            self._table_rows = {}
        self._codes = {}
        self.nested = value.get('nested', {})
        self._fixed_bytes = {row: bytes_of(atom[1]) for atom, row in self.atoms.items()}
        self._fixed_bytes.update({row: raw for raw, row in self.word_rows.items()})
        self._atom_rows, self._catalog = {}, {}
        if 'holders' in value and 'inside' in value:
            self.holders = {word: set(found) for word, found in value['holders'].items()}
            self.inside = {word: set(found) for word, found in value['inside'].items()}
        else:
            self._relink()
        self.order = {word: index for index, word in enumerate(self.words)}
        for word, row in self.word_rows.items():
            if row not in self.store.part_groups:
                self.retain_spelling(word)

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
        # Forms may share a parts half: a form is known by its row. Reported, not an error.
        return dict(vocabulary=[text_of(w) for w in words], pair_dim=self.pair_dim,
            ones=self.ones,
            shared_codes=len(words)-len({tuple(v.tolist()) for v in forms.values()}),
            containment_pairs=[(text_of(a), text_of(b)) for a, b in comparable],
            containment_violations=sum(bool((forms[a] > forms[b]).any()) for a,b in comparable),
            reconstruction_errors=sum(self.spelling(self.parts(w)) != w for w in words), added=self.added)
