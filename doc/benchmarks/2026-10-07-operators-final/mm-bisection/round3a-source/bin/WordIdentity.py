"""Rung-zero identity: boundary pairs, exact length parts, collision mints.

Atom rows and word rows belong to the existing RadixLayer. Their codes are
fixed functions of atom bytes, not parameters. Only a recorded collision
may extend a minted atom's deterministic bit stream. Binding uses a separate
fixed dense view; indexes and reconstruction retain the sparse form.
"""
import hashlib
from functools import lru_cache

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def resolve_identity_layout(config):
    """Resolve text form storage before constructing any spaces.

    Older XML files described a learned letter width. Identity now owns an
    explicit pair width plus 32 length bits. Preserve the declared meaning
    complement and live slot/row counts; numeric configurations are untouched.
    """
    from architecture import canonical_shape
    arch = config.get('architecture', {})
    if arch.get('data', {}).get('dataType', arch.get('dataType')) != 'embedding':
        return
    ps = config.setdefault('PartSpace', {})
    if not ps.get('identityByParts', True):
        return
    pair = int(ps.setdefault('identityPairDim', 64))
    ones = int(ps.setdefault('identityOnes', 3))
    length = int(ps.setdefault('identityLengthDim', 32))
    dense = int(ps.setdefault('identityBindingDim', 64))
    if pair < ones or ones < 1 or length != 32 or dense < 1:
        raise ValueError('identity requires D >= s > 0, L = 32 and a positive binding width')
    width = pair + length
    if dense > width:
        raise ValueError('binding width must fit the reserved form block')
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
    word = text_of(raw)
    marked = '#' + word + '#'
    return tuple(dict.fromkeys(
        [('pair', marked[i:i+2]) for i in range(len(marked)-1)] +
        [('length', str(i)) for i in range(1, len(word)+1)]))


class WordIdentity(nn.Module):
    PREFIX = b'\x00identity/atom/'
    WORD_PREFIX = b'\x00identity/form/'

    def __init__(self, store, pair_dim=64, ones=3, length_dim=32, binding_dim=64):
        super().__init__()
        object.__setattr__(self, 'store', store)
        self.pair_dim, self.ones = int(pair_dim), int(ones)
        self.length_dim, self.binding_dim = int(length_dim), int(binding_dim)
        self.width = self.pair_dim + self.length_dim
        if store.dim != self.width or not 0 < self.ones <= self.pair_dim:
            raise ValueError('Radix identity bank width must be D + L, with 0 < s <= D')
        seed = int.from_bytes(hashlib.sha256(b'word-identity-projection-v1').digest()[:8], 'little')
        # A local generator cannot move any training's initialization stream.
        matrix = np.random.default_rng(seed).normal(size=(self.width, self.binding_dim))
        self.register_buffer('projection', torch.tensor(matrix / self.binding_dim**.5, dtype=torch.float32))
        self.atoms, self.bits, self.words, self.word_rows, self.forms = {}, {}, {}, {}, {}
        self.word_keys = {}
        self.mints = []
        self.revision = 0
        basis = store._basis
        if not hasattr(basis, 'fixed_row_mask'):
            basis.register_buffer('fixed_row_mask', torch.zeros(store._capacity, dtype=torch.bool, device=basis.W.device))
            basis.register_buffer('fixed_row_codes', torch.zeros_like(basis.W, requires_grad=False))

    def atom_code(self, atom, bits=None):
        kind, value = atom
        result = torch.zeros(self.width)
        if kind == 'length':
            k = int(value)
            if k <= self.length_dim:
                result[self.pair_dim+k-1] = 1
        else:
            n = self.bits.get(atom, self.ones) if bits is None else int(bits)
            result[list(bit_stream(bytes_of(value), self.pair_dim)[:n])] = 1
        return result

    @torch.no_grad()
    def _fixed(self, row, value):
        basis = self.store._basis
        value = value.to(basis.W)
        basis.fixed_row_mask[row] = True
        basis.fixed_row_codes[row].copy_(value)
        basis.W[row].copy_(value)

    def _atom(self, atom, bits=None):
        if bits is not None:
            self.bits[atom] = max(self.bits.get(atom, self.ones), int(bits))
        code = self.atom_code(atom)
        if atom not in self.atoms:
            key = self.PREFIX + atom[0].encode() + b'/' + bytes_of(atom[1])
            self.atoms[atom] = self.store.insert(key, init_vector=code)
        self._fixed(self.atoms[atom], code)
        return self.atoms[atom]

    def form(self, raw):
        atoms = self.words.get(bytes(raw), base_atoms(raw))
        return torch.stack([self.atom_code(a) for a in atoms]).amax(0)

    @staticmethod
    def key(form):
        return np.packbits(form.detach().cpu().numpy().astype(np.uint8), bitorder='little').tobytes()

    def _mint(self, left, right):
        a, b = '#' + text_of(left) + '#', '#' + text_of(right) + '#'
        old_a, old_b = self.form(left), self.form(right)
        positions = [pos for pos in range(max(len(a),len(b))-2)
                     if len(a[pos:pos+3]) == len(b[pos:pos+3]) == 3
                     and a[pos:pos+3] != b[pos:pos+3]]
        # Exhaust the differing triple positions, then try quadruples at
        # those positions. Unsuccessful proposals never alter stored atoms.
        for size in (3, 4):
            for pos in positions:
                x, y = a[pos:pos+size], b[pos:pos+size]
                if len(x) != size or len(y) != size or x == y:
                    continue
                aa, bb = ('mint', f'{x}@{pos}'), ('mint', f'{y}@{pos}')
                for bits in range(max(self.bits.get(aa, self.ones), self.bits.get(bb, self.ones)), self.pair_dim+1):
                    ca, cb = self.atom_code(aa, bits), self.atom_code(bb, bits)
                    if not torch.equal(torch.maximum(old_a, ca), torch.maximum(old_b, cb)):
                        self._atom(aa, bits); self._atom(bb, bits)
                        self.words[left] = tuple(dict.fromkeys((*self.words[left], aa)))
                        self.words[right] = tuple(dict.fromkeys((*self.words[right], bb)))
                        self.mints.append(dict(words=[text_of(left), text_of(right)],
                            atoms=[aa[1], bb[1]], bit_counts=[bits, bits], size=size, position=pos))
                        self.revision += 1
                        return
        raise ValueError(f'identity mint exhausted triples and quadruples: {left!r}, {right!r}')

    def _reindex(self):
        self.forms = {}
        self.word_keys = {}
        for raw in self.words:
            key = self.key(self.form(raw))
            if key in self.forms and self.forms[key] != raw:
                return self.forms[key], raw
            self.forms[key] = raw
            self.word_keys[raw] = key
        return None

    def admit(self, raw):
        raw = bytes(raw)
        if not raw:
            raise ValueError('a word identity requires nonempty bytes')
        if raw not in self.words:
            self.words[raw] = base_atoms(raw)
            for atom in self.words[raw]:
                self._atom(atom)
            pending, affected = [raw], {raw}
            while pending:
                word = pending.pop()
                key = self.key(self.form(word))
                other = self.forms.get(key)
                if other is not None and other != word:
                    self._mint(other, word)
                    changed = set(self.words[other][-1:]) | set(self.words[word][-1:])
                    touched = {w for w, parts in self.words.items() if changed.intersection(parts)}
                    affected.update(touched)
                    for w in touched:
                        old = self.word_keys.pop(w, None)
                        if old is not None and self.forms.get(old) == w:
                            del self.forms[old]
                            self.store.hash_map.pop(self.WORD_PREFIX+old, None)
                    pending.extend(touched)
                else:
                    self.forms[key] = word
                    self.word_keys[word] = key
            # Rekey the affected old words too; their row addresses are stable.
            for word in sorted(affected):
                value = self.form(word)
                key = self.WORD_PREFIX + self.key(value)
                row = self.word_rows.get(word)
                if row is None:
                    row = self.store.insert(key, init_vector=value)
                    self.word_rows[word] = row
                    self.store.inverse_table[row] = word
                    # Keep byte rows distinct for one-letter words/descent.
                    if len(word) > 1:
                        self.store.hash_map[word] = row
                        self.store.radix_trie.insert(word, row)
                self._fixed(row, value)
                self.store.hash_map[key] = row
                self.store.radix_trie.insert(key, row)
        return tuple(self.atoms[a] for a in self.words[raw])

    def row_for_form(self, form):
        raw = self.forms.get(self.key(form))
        return None if raw is None else self.word_rows[raw]

    def read(self, form):
        row = self.row_for_form(form)
        return None if row is None else self.store.bytes_for(row)

    def read_parts(self, rows, *, form=None):
        """Read an indexed identity, or assemble an unindexed pair spelling."""
        atoms = {row: atom for atom, row in self.atoms.items()}
        parts = [atoms[int(row)] for row in rows]
        if form is None:
            form = torch.stack([self.atom_code(atom) for atom in parts]).amax(0)
        known = self.read(form)
        if known is not None:
            return known
        lengths = [int(value) for kind, value in parts if kind == 'length']
        if not lengths:
            raise ValueError('pair reconstruction requires cumulative length parts')
        return assemble_pairs([value for kind, value in parts if kind == 'pair'],
                              max(lengths), index=self, form=form)

    def binding(self, atoms):
        """Preserve the event/meaning layout while changing only code direction."""
        projected = atoms[..., :self.width] @ self.projection.to(atoms)
        return torch.cat((F.pad(projected, (0, self.width-self.binding_dim)),
                          atoms[..., self.width:]), -1)

    def vocab_extras(self):
        return dict(atoms=self.atoms, bits=self.bits, words=self.words,
                    word_rows=self.word_rows, mints=self.mints, revision=self.revision)

    def load_vocab_extras(self, value):
        for name in ('atoms', 'bits', 'words', 'word_rows', 'mints', 'revision'):
            setattr(self, name, value[name])
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
            length_dim=self.length_dim, ones=self.ones, binding_dim=self.binding_dim,
            collisions=len(words)-len({self.key(v) for v in forms.values()}),
            containment_pairs=[(text_of(a), text_of(b)) for a, b in comparable],
            containment_violations=sum(bool((forms[a] > forms[b]).any()) for a,b in comparable),
            reconstruction_errors=sum(self.read(forms[w]) != w for w in words), mints=self.mints)


def assemble_pairs(pairs, length, *, index=None, form=None):
    """Assemble an unindexed spelling from its pair set and exact length.

    Pair sets omit multiplicity. Ambiguous sequences require the minted
    identity index; no arbitrary spelling is substituted for an identity.
    """
    if index is not None and form is not None:
        known = index.read(form)
        if known is not None:
            return known
    pairs = set(pairs)
    answers = []
    def walk(text, used):
        if len(answers) > 1:
            return
        if len(text) == length + 1:
            end = text[-1]+'#'
            if end in pairs and used | {end} == pairs:
                answers.append(bytes_of(text[1:]))
            return
        for pair in sorted(pairs):
            if pair[0] == text[-1] and pair[1] != '#':
                walk(text+pair[1], used | {pair})
    walk('#', set())
    if len(answers) != 1:
        raise ValueError('pair/length spelling is unresolved; a minted index identity is required')
    return answers[0]
