"""The trial of the balanced code of a basic percept, in tests only.

sigma-pi §19.69 and §22.3b (and Alec's further answers, §19.70). Nothing in
bin/ is used to build the codes tried here; the source's own pieces are
imported only where the trial stands in for them (``bit_stream``,
``distinguishing_run``, ``feeling``, the byte ranges of the declared
properties, ``combine``). Each coder keeps its codes as Python integers, one
bit a coordinate, so that a dictionary of 19,821 words is coded in seconds.

Layouts:

- ``built``: the code as built, 352 wide. The parts half is 96 (parts drawn
  in the first 64, 32 empty); the wholes half is 256, a whole coded by the
  bytes it holds on. Two forms are compared on the parts half only.
- ``balanced``: two halves of one width ``n``. A part has 3 coordinates of
  its own in the parts half, a whole 3 in the wholes half; a percept's
  wholes half is all ones less the coordinates of the wholes it lies in
  (P2). Two forms are compared on both halves. With ``alternate`` a
  collision is resolved by a new part or a new whole in turn (P4).

The signed code (P6) and the scalar table are functions over a coder.
"""
import hashlib
import time
import math
import os
from collections import Counter

import numpy as np
import torch

from WordIdentity import bit_stream, distinguishing_run, runs
from Layers import _CHAR_CLASS_RANGES, LETTER, DIGIT, WHITESPACE, PUNCT, CAPITAL, CONTROL, HIGH_BYTE, PAD

ONES = 3


# ---------------------------------------------------------------------------
# Inputs.

def dictionary(count=20000, path='/usr/share/dict/words'):
    """The first ``count`` words of the dictionary that are letters only,
    2 to 12 long, in lower case; each once, in order (19,821)."""
    with open(path) as handle:
        words = [w.lower() for w in handle.read().split()
                 if w.isascii() and w.isalpha() and 2 <= len(w) <= 12][:count]
    return list(dict.fromkeys(words))


def declared_wholes():
    """The nine declared properties and the bytes each holds on (the
    canonical rows of WholeSpace; "word" is the letters, ``define_word``)."""
    def span(cls):
        return frozenset(b for lo, hi in _CHAR_CLASS_RANGES[cls] for b in range(lo, hi + 1))
    names = (('letter', LETTER), ('digit', DIGIT), ('whitespace', WHITESPACE),
             ('punctuation', PUNCT), ('capital', CAPITAL), ('control', CONTROL),
             ('high byte', HIGH_BYTE), ('pad', PAD))
    result = {name: span(cls) for name, cls in names}
    result['word'] = result['letter']
    return result


def drawn(count=1):
    """Starts drawn from the system, to be printed."""
    return [int.from_bytes(os.urandom(4), 'little') for _ in range(count)]


def coords(name, width, ones=ONES):
    """The first ``ones`` coordinates of a name's stream of bits."""
    return tuple(bit_stream(name if isinstance(name, bytes) else name.encode(), width)[:ones])


def mask_of(indices):
    m = 0
    for i in indices:
        m |= 1 << i
    return m


def bits(mask, width):
    """A mask as a 0/1 vector."""
    return np.array([(mask >> i) & 1 for i in range(width)], dtype=np.float64)


def popcount(mask):
    return bin(mask).count('1')


# ---------------------------------------------------------------------------
# The coder.

class Coder:
    """Forms admitted in order, each with a code; collisions resolved as
    ``WordIdentity._settle`` and ``_separate`` resolve them (as built), or
    by a part or a whole in turn (P4)."""

    def __init__(self, layout='balanced', n=64, *, alternate=False, wholes=None):
        self.layout, self.alternate = layout, bool(alternate)
        if layout == 'built':
            self.parts_n, self.empty, self.wholes_n = 64, 32, 256
        else:
            self.parts_n, self.empty, self.wholes_n = int(n), 0, int(n)
        self.bits = {}                # atom -> ones, where raised
        self._atom_masks = {}
        self.added = {}               # run -> the two forms it told apart
        self.held = Counter()         # added atom -> forms holding it
        self.order, self.words, self.parts = {}, [], {}
        self.forms, self.keys, self.shared = {}, {}, {}
        self.wholes = dict(declared_wholes() if wholes is None else wholes)
        self.kinds = {name: 'declared' for name in self.wholes}
        self.whole_of = {}            # form -> names of the wholes it lies in
        self.made = []                # (kind, name or run, left, right)
        self.collisions = []          # (left, right, bytes differ, outcome)
        self._whole_masks = {}

    # Codes ---------------------------------------------------------------

    def atom_mask(self, atom, ones=None):
        if ones is None and atom in self._atom_masks:
            return self._atom_masks[atom]
        k = self.bits.get(atom, ONES) if ones is None else ones
        m = mask_of(coords(atom, self.parts_n, k))
        if ones is None:
            self._atom_masks[atom] = m
        return m

    def part_mask(self, form, override=None):
        m = 0
        for atom in self.parts[form]:
            m |= self.atom_mask(atom, override[1] if override and atom == override[0] else None)
        return m

    def whole_own(self, name):
        """A whole's own coordinates in the wholes half (balanced)."""
        if name not in self._whole_masks:
            self._whole_masks[name] = mask_of(coords(b'\x00whole/' + name.encode(), self.wholes_n))
        return self._whole_masks[name]

    def ancestors(self, name):
        """The wholes a whole lies in: those holding on all of its bytes."""
        own = self.wholes[name]
        return tuple(other for other, held in self.wholes.items() if own <= held)

    def whole_code_mask(self, name):
        """A whole's code in the wholes half: the coordinates it is not."""
        if self.layout == 'built':
            return mask_of(self.wholes[name])
        removed = 0
        for other in self.ancestors(name):
            removed |= self.whole_own(other)
        return removed

    def wholes_for(self, form):
        held = set(form.encode())
        return frozenset(name for name, members in self.wholes.items() if held <= members)

    def wholes_mask(self, form):
        """Built: the least of its wholes' byte sets (ones). Balanced: the
        coordinates removed from all ones (a set bit is a zero)."""
        names = self.whole_of[form]
        if self.layout == 'built':
            m = (1 << 256) - 1
            for name in names:
                m &= mask_of(self.wholes[name])
            return m
        m = 0
        for name in names:
            m |= self.whole_own(name)
        return m

    def wholes_vector(self, form):
        if self.layout == 'built':
            return bits(self.wholes_mask(form), 256)
        return 1. - bits(self.wholes_mask(form), self.wholes_n)

    def parts_vector(self, form):
        return np.concatenate((bits(self.part_mask(form), self.parts_n), np.zeros(self.empty)))

    def vector(self, form):
        return np.concatenate((self.parts_vector(form), self.wholes_vector(form)))

    def key(self, form):
        if self.layout == 'built':
            return self.part_mask(form)
        return (self.part_mask(form), self.wholes_mask(form))

    # Admission -----------------------------------------------------------

    def atoms_of(self, form):
        return frozenset(dict.fromkeys(form)) | frozenset(run for run in self.added if run in form)

    def admit(self, form):
        if form in self.order:
            return
        self.order[form] = len(self.words)
        self.words.append(form)
        self.parts[form] = self.atoms_of(form)
        for atom in self.parts[form]:
            if atom in self.added:
                self.held[atom] += 1
        self.whole_of[form] = self.wholes_for(form)
        carriers = []
        if self.layout == 'built' and len(form) > 1 and form not in self.added:
            # As built, rule 43 as amended: a stored form of several
            # characters is a part of its own, carried by its holders.
            carriers = [w for w in self._add_part(form, ('own', form)) if w != form]
        self._settle(carriers + [form])

    def admit_all(self, forms):
        for form in forms:
            self.admit(form)
        return self

    def _unkey(self, form):
        old = self.keys.pop(form, None)
        if old is not None and self.forms.get(old) == form:
            del self.forms[old]

    def _settle(self, pending):
        pending = list(pending)
        while pending:
            word = pending.pop()
            self._unkey(word)
            self.shared.pop(word, None)
            key = self.key(word)
            other = self.forms.get(key)
            if other is not None and other != word:
                touched = self._separate(other, word)
                if touched is None:
                    if self.order[word] < self.order[other]:
                        self.shared.pop(word, None)
                        self.shared[other] = word
                        self.forms[key] = word
                    self.keys[word] = key
                    continue
                self._unkey(other)
                again = sorted(set(touched) | {word, other}, key=self.order.__getitem__)
                pending.extend(again)
            else:
                self.forms[key] = word
                self.keys[word] = key

    def _add_part(self, run, between):
        self.added[run] = between
        holders = []
        for word in self.words:
            if run in word:
                self.parts[word] = self.parts[word] | {run}
                holders.append(word)
        self.held[run] = len(holders)
        return holders

    def _part_step(self, left, right):
        """WordIdentity._separate's two steps: a new part where the parts are
        the same, then more ones for an added part one holds alone."""
        holders = []
        if self.parts[left] == self.parts[right]:
            run = distinguishing_run(left.encode(), right.encode())
            holders = self._add_part(run, (left, right))
            self.made.append(('part', run, left, right))
            if self.key(left) != self.key(right):
                return holders
        candidates = sorted((a for a in self.parts[left] ^ self.parts[right] if a in self.added),
                            key=lambda a: (self.held[a], -len(a), a))
        for atom in candidates:
            for k in range(self.bits.get(atom, ONES) + 1, self.parts_n + 1):
                if self.part_mask(left, (atom, k)) != self.part_mask(right, (atom, k)):
                    self.bits[atom] = k
                    self._atom_masks.pop(atom, None)
                    self.made.append(('bits', atom, left, right))
                    return holders + [w for w in self.words if atom in self.parts[w]]
        return holders or None

    def _whole_step(self, left, right):
        """A new property taught the bytes of the form admitted first, or of
        the other where the first's bytes hold all of the other's."""
        a, b = frozenset(left.encode()), frozenset(right.encode())
        if a == b:
            return None
        taught = a if not b <= a else b
        name = 'made/%d' % len(self.wholes)
        self.wholes[name] = taught
        self.kinds[name] = 'made'
        holders = [w for w in self.words if frozenset(w.encode()) <= taught]
        for w in holders:
            self.whole_of[w] = self.whole_of[w] | {name}
        self.made.append(('whole', name, left, right))
        return holders

    def _separate(self, left, right):
        differ = frozenset(left.encode()) != frozenset(right.encode())
        steps = [self._part_step]
        if self.alternate:
            wholes = sum(1 for m in self.made if m[0] == 'whole')
            parts = sum(1 for m in self.made if m[0] != 'whole')
            steps = [self._whole_step, self._part_step] if wholes <= parts else [self._part_step, self._whole_step]
        touched = None
        for step in steps:
            found = step(left, right)
            if found is not None:
                touched = (touched or []) + found
                if self.key(left) != self.key(right):
                    self.collisions.append((left, right, differ, step.__name__))
                    return touched
        self.collisions.append((left, right, differ, 'shared'))
        self.shared[right] = left
        return touched

    # Reports -------------------------------------------------------------

    def summary(self):
        made = Counter(m[0] for m in self.made)
        pairs = Counter((differ, outcome) for _l, _r, differ, outcome in self.collisions)
        p = [popcount(self.part_mask(w)) for w in self.words]
        if self.layout == 'built':
            wh = [popcount(self.wholes_mask(w)) for w in self.words]
        else:
            wh = [self.wholes_n - popcount(self.wholes_mask(w)) for w in self.words]
        return dict(
            words=len(self.words), sharing=len(self.shared),
            parts_made=made['part'], bits_raised=made['bits'], wholes_made=made['whole'],
            collided_pairs=len(self.collisions),
            pairs_bytes_differ=sum(v for (d, _o), v in pairs.items() if d),
            pairs_bytes_same=sum(v for (d, _o), v in pairs.items() if not d),
            outcomes={'%s/%s' % ('bytes differ' if d else 'same bytes', o): v for (d, o), v in sorted(pairs.items())},
            parts_ones=(min(p), round(float(np.mean(p)), 2), max(p)),
            wholes_ones=(min(wh), round(float(np.mean(wh)), 2), max(wh)))

    def matrices(self):
        """Every form's two halves as 0/1 boolean arrays (rows in order)."""
        P = np.array([[(self.part_mask(w) >> i) & 1 for i in range(self.parts_n)] for w in self.words], dtype=bool)
        W = np.array([self.wholes_vector(w) > .5 for w in self.words], dtype=bool)
        return P, W

    def accidents(self, sample_every=40):
        """For each sampled form against all: codes at or below another's,
        truly (the parts inside; the other's wholes among its own) and by
        accident, in each half and in the whole code."""
        P, W = self.matrices()
        parts = [self.parts[w] for w in self.words]
        wholes = [self.whole_of[w] for w in self.words]
        result = {half: [0, 0] for half in ('parts', 'wholes', 'both')}
        sample = list(range(0, len(self.words), sample_every))
        for i in sample:
            p_le = (~P[i] | P).all(1)
            w_le = (~W[i] | W).all(1)
            p_le[i] = w_le[i] = False
            for half, le, true in (('parts', p_le, lambda j: parts[i] <= parts[j]),
                                   ('wholes', w_le, lambda j: wholes[j] <= wholes[i]),
                                   ('both', p_le & w_le,
                                    lambda j: parts[i] <= parts[j] and wholes[j] <= wholes[i])):
                for j in np.flatnonzero(le):
                    result[half][0 if true(j) else 1] += 1
        return dict(sample=len(sample), **{h: dict(true=t, accident=a) for h, (t, a) in result.items()})


# ---------------------------------------------------------------------------
# The signed code (P6).

def holders_of(words):
    """For each form, the stored forms that hold it as a run (its wholes as
    forms; FormContainment by the run, not by the set of parts)."""
    stored = set(words)
    result = {w: set() for w in words}
    for w in words:
        for size in range(2, len(w)):
            for start in range(len(w) - size + 1):
                run = w[start:start + size]
                if run in stored:
                    result[run].add(w)
    return result


def signed(coder, holders=None, *, broad=False):
    """Each form's signed code, both halves. Parts: +1 at its parts'
    coordinates, -1 at those its holders add (or, ``broad``, at those of
    every part in the catalog it does not hold), 0 elsewhere; +1 stands
    where a coordinate is both. Wholes: +1 at the coordinates of the wholes
    that apply, -1 at those of the wholes witnessed not to apply (every
    whole is over bytes, so each is witnessed for a form). Returns the
    codes and the count of coordinates that were both."""
    if coder.layout == 'built':
        raise ValueError('the signed code is tried in the balanced layout')
    n, m = coder.parts_n, coder.wholes_n
    catalog = set()
    for w in coder.words:
        catalog |= coder.parts[w]
    catalog_masks = {a: coder.atom_mask(a) for a in catalog}
    codes = np.zeros((len(coder.words), n + m))
    both_parts = both_wholes = 0
    for row, w in enumerate(coder.words):
        own = coder.part_mask(w)
        against = 0
        if broad:
            for atom, mk in catalog_masks.items():
                if atom not in coder.parts[w]:
                    against |= mk
        elif holders is not None:
            for h in holders.get(w, ()):
                # The atoms a holder adds, at their coordinates.
                for atom in coder.parts[h] - coder.parts[w]:
                    against |= coder.atom_mask(atom)
        both_parts += popcount(against & own)
        codes[row, :n] = bits(own, n) - bits(against & ~own, n)
        applies, fails = 0, 0
        for name in coder.wholes:
            if name in coder.whole_of[w]:
                applies |= coder.whole_own(name)
            else:
                fails |= coder.whole_own(name)
        both_wholes += popcount(applies & fails)
        codes[row, n:] = bits(applies, m) - bits(fails & ~applies, m)
    return codes, dict(both_in_parts=both_parts, both_in_wholes=both_wholes)


def lanes(code):
    """A signed code as its two unsigned lanes, for and against."""
    return np.clip(code, 0, None), np.clip(-code, 0, None)


def columns(codes):
    """Rule 8b: each concept's column, its member's code over its squared
    length (signed codes as they are)."""
    size = (codes ** 2).sum(1, keepdims=True)
    return codes / np.where(size > 0, size, 1.)


def readings(codes, cols=None):
    """Before the squash: row ``i`` (a reading of form ``i`` alone) read by
    concept ``j``. A signed code reads its for lane less its against lane,
    which for one percept at presence one is the signed code itself."""
    cols = columns(codes) if cols is None else cols
    return codes @ cols.T


def told_apart(R):
    """As ``test_how_many_concepts_are_told_apart``: a reading is told apart
    when its own concept reads it above every other concept."""
    y = np.tanh(R)
    own = np.diag(y).copy()
    other = y - np.diag(own) - np.eye(len(y)) * 2
    # A reading another concept takes at its own (to rounding) is a tie, not
    # told apart.
    apart = int((own > other.max(1) + 1e-9).sum())
    ties = int((np.abs(own - other.max(1)) <= 1e-9).sum())
    off = R[~np.eye(len(R), dtype=bool)]
    return dict(told_apart=apart, of=len(R), ties=ties, own=round(float(np.tanh(np.diag(R)).mean()), 3),
                others_mean_before_squash=round(float(off.mean()), 3),
                others_mean=round(float(np.tanh(off).mean()), 3),
                others_greatest=round(float(np.tanh(off.max())), 3))


# ---------------------------------------------------------------------------
# Centroids.

def auc(member_scores, other_scores):
    """The chance that a member scores above a non-member (ties half)."""
    a, b = np.asarray(member_scores), np.asarray(other_scores)
    if not len(a) or not len(b):
        return float('nan')
    order = np.concatenate((a, b)).argsort(kind='mergesort')
    ranks = np.empty(len(order))
    values = np.concatenate((a, b))[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[j + 1] == values[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b)))


def centroid_report(codes, members, split):
    """The centroid of a class, what share of its mass each half gives,
    the share of that mass that is not the centroid of everything, and
    whether members are nearer it than non-members: by the share of the
    centroid a code holds (``sum min(v, c) / sum c``, for codes of 0 and 1;
    ``v . c / c . c`` for signed codes) and by cosine (AUC, members over
    non-members)."""
    members = np.asarray(sorted(members))
    mask = np.zeros(len(codes), dtype=bool)
    mask[members] = True
    c = codes[mask].mean(0)
    everything = codes.mean(0)
    mass = np.abs(c).sum()
    signed_codes = bool((codes < 0).any())
    if signed_codes:
        held = codes @ c / max(float(c @ c), 1e-12)
    else:
        held = np.minimum(codes, c).sum(1) / max(float(c.sum()), 1e-12)
    norms = np.linalg.norm(codes, axis=1) * max(float(np.linalg.norm(c)), 1e-12)
    cosine = codes @ c / np.where(norms > 0, norms, 1.)
    return dict(size=int(mask.sum()),
                parts_share=round(float(np.abs(c[:split]).sum() / mass), 3),
                wholes_share=round(float(np.abs(c[split:]).sum() / mass), 3),
                not_common=round(float(np.abs(c - everything).sum() / mass), 3),
                held_member=round(float(held[mask].mean()), 3), held_other=round(float(held[~mask].mean()), 3),
                held_auc=round(auc(held[mask], held[~mask]), 3),
                cosine_member=round(float(cosine[mask].mean()), 3), cosine_other=round(float(cosine[~mask].mean()), 3),
                cosine_auc=round(auc(cosine[mask], cosine[~mask]), 3))


# ---------------------------------------------------------------------------
# Synthetic wholes for point 2.

def nested_wholes(count, seed, base=None):
    """``count`` wholes over bytes: the declared nine, then each new one a
    random half of a random earlier one's bytes (so some lie in others)."""
    rng = np.random.default_rng(seed)
    result = dict(declared_wholes() if base is None else base)
    names = list(result)
    while len(result) < count:
        parent = result[names[int(rng.integers(len(names)))]]
        if len(parent) < 2:
            continue
        members = sorted(parent)
        chosen = frozenset(rng.choice(members, size=max(1, len(members) // 2), replace=False).tolist())
        name = 'w%d' % len(result)
        result[name] = chosen
        names.append(name)
    return result


def whole_order(wholes, n, ones=ONES):
    """P2 for wholes alone: each whole's code is all ones less its own and
    its ancestors' coordinates. Returns how many truly-ordered pairs are at
    or below, how many fail to be (must be 0), the accidents, and how many
    distinct byte sets share one code."""
    names = list(wholes)
    own = {name: mask_of(coords(b'\x00whole/' + name.encode(), n, ones)) for name in names}
    removed = {}
    for name in names:
        r = 0
        for other in names:
            if wholes[name] <= wholes[other]:
                r |= own[other]
        removed[name] = r
    true = failed = accident = 0
    for a in names:
        for b in names:
            if a == b:
                continue
            at_or_below = (removed[b] & ~removed[a]) == 0
            inside = wholes[a] <= wholes[b]
            if inside:
                true += 1
                failed += not at_or_below
            elif at_or_below:
                accident += 1
    codes = Counter(removed.values())
    sets = Counter(frozenset(wholes[name]) for name in names)
    return dict(wholes=len(names), true=true, not_ordered=failed, accidents=accident,
                pairs=len(names) * (len(names) - 1),
                codes_shared=sum(v - 1 for v in codes.values() if v > 1),
                byte_sets_shared=sum(v - 1 for v in sets.values() if v > 1))


# ---------------------------------------------------------------------------
# Scalars.

def felt(value):
    return float(math.log1p(value) / (1 + math.log1p(value)))


def read_matrix(inputs, cols):
    """``inputs`` [cases, 2, W] (for and against lanes), ``cols`` [K, W]."""
    return np.tanh((inputs[:, 0] - inputs[:, 1]) @ cols.T)


def combine_lanes(present):
    """Lanes of several percepts at presence one: 1 - prod(1 - v) on each."""
    if not present:
        return None
    f = 1 - np.prod([1 - lanes(c)[0] for c in present], 0)
    a = 1 - np.prod([1 - lanes(c)[1] for c in present], 0)
    return np.stack((f, a))


class ValueTable:
    """Alec's scalar code (§19.70, point 3): one sorted table of the values
    seen; a value's parts are the values at or below it (+1), its wholes the
    values above it (-1). ``mode`` gives the table's entries coordinates in
    ``width``: 'exact' (one each; does not scale), 'bins' (the bin of the
    felt value, ``floor(width * felt)``; a value shares its bin's
    coordinate), or 'hashed' (3 coordinates drawn from the value's name)."""

    def __init__(self, width=64, mode='exact'):
        self.width, self.mode, self.values = int(width), mode, []

    def see(self, value):
        if value not in self.values:
            self.values.append(value)
            self.values.sort()

    def place(self, value):
        if self.mode == 'exact':
            return (self.values.index(value),)
        if self.mode == 'bins':
            return (min(self.width - 1, int(self.width * felt(value))),)
        return coords(b'\x00value/' + repr(float(value)).encode(), self.width)

    def code(self, value):
        self.see(value)
        width = len(self.values) if self.mode == 'exact' else self.width
        result = np.zeros(max(width, self.width if self.mode != 'exact' else width))
        for other in self.values:
            for i in self.place(other):
                if other <= value:
                    result[i] = 1.
                elif result[i] == 0:
                    result[i] = -1.
        return result


def pad(vectors, width):
    return np.stack([np.pad(v, (0, width - len(v))) for v in vectors])


# ---------------------------------------------------------------------------
# The first Pi with entries on both lanes (§19.71, point 2).

def against_entries(coder, holders):
    """A row's against entries: the parts that the forms holding it as a
    run add to it (``y`` of ``archery`` and ``s`` of ``archers`` for
    ``archer``)."""
    return {w: frozenset(a for h in holders.get(w, ()) for a in coder.parts[h] - coder.parts[w])
            for w in coder.words}


def pi_row(coder, row, reading, against, *, join, by='set'):
    """The two lanes of ``row``'s basic percept on a reading of the form
    ``reading``. A member's membership is 0 or 1: by the reading's set of
    parts (``set``) or by its code holding the member's code (``code``).
    For: the product of the row's parts. Against: the run's presence times
    the join of the against entries, their product (``product``: every one
    present) or their union (``union``, 1 - prod(1 - m): any one)."""
    if by == 'set':
        parts = coder.parts[reading]
        held = parts.__contains__
    else:
        mask = coder.part_mask(reading)
        held = lambda atom: (coder.atom_mask(atom) & ~mask) == 0
    present = float(all(held(a) for a in coder.parts[row]))
    entries = against[row]
    if not entries:
        return present, 0.
    found = [held(a) for a in entries]
    joined = all(found) if join == 'product' else any(found)
    return present, present * float(joined)


def graded_table_code(table, value, grade):
    """The sorted table's two lanes for ``value``, with a value's evidence
    graded by its distance on the felt chart: ``against`` grades the against
    lane at an entry above, ``1 - felt(v) / felt(t)``; ``both`` also grades
    the for lane at an entry below, ``felt(t) / felt(v)``. Returns the for
    lane less the against lane (the two are never both above zero at one
    entry)."""
    crisp = table.code(value)
    result = crisp.copy()
    for other in table.values:
        for i in table.place(other):
            if other > value and grade in ('against', 'both') and crisp[i] < 0:
                result[i] = -(1 - felt(value) / felt(other))
            elif other <= value and grade == 'both' and crisp[i] > 0:
                result[i] = max(result[i] if result[i] < 1 else 0., felt(other) / felt(value))
    return result


# ---------------------------------------------------------------------------
# The wholes half as an upper bound, read on its complement (§19.72).

TOP = 'all characters'


def with_top(wholes):
    """The wholes and the class of all characters above them."""
    result = dict(wholes)
    result[TOP] = frozenset(range(256))
    return result


def children(wholes):
    """Each whole's wholes directly inside it (strictly fewer bytes, none
    between); wholes of one byte set (letter, word) are one level."""
    sets = {name: wholes[name] for name in wholes}
    result = {}
    for a, A in sets.items():
        below = [b for b, B in sets.items() if B < A]
        result[a] = tuple(b for b in below
                          if not any(sets[b] < sets[c] < A for c in below))
    return result


def complement_code(coder, form, holders=None, *, kids=None):
    """A form's two lanes with the wholes half on its complement.

    Parts half: for at its parts' coordinates; against, with ``holders``,
    at the coordinates its holders add (P6), else none. Wholes half: for at
    the coordinates its wholes cut away from all ones (the class of all
    characters cuts none); against at the own coordinates of the wholes
    directly inside those that apply which do not apply. Returns
    ``(for, against)``, each ``2n`` wide in 0 to 1, and the names of the
    wholes on the against lane."""
    n, m = coder.parts_n, coder.wholes_n
    wholes = with_top(coder.wholes)
    kids = children(wholes) if kids is None else kids
    applies = set(coder.whole_of[form]) | {TOP}
    own = coder.part_mask(form)
    against_parts = 0
    if holders is not None:
        for h in holders.get(form, ()):
            for atom in coder.parts[h] - coder.parts[form]:
                against_parts |= coder.atom_mask(atom)
    cut = 0
    for name in applies - {TOP}:
        cut |= coder.whole_own(name)
    witnessed = sorted({b for a in applies for b in kids[a] if b not in applies})
    against_wholes = 0
    for name in witnessed:
        against_wholes |= coder.whole_own(name)
    f = np.concatenate((bits(own, n), bits(cut, m)))
    a = np.concatenate((bits(against_parts & ~own, n), bits(against_wholes & ~cut, m)))
    return f, a, witnessed


def complement_codes(coder, holders=None):
    kids = children(with_top(coder.wholes))
    pairs = [complement_code(coder, w, holders, kids=kids) for w in coder.words]
    return np.stack([f - a for f, a, _ in pairs]), pairs


# ---------------------------------------------------------------------------
# Whether the two-sided code replaces the Not at the basic level (§19.72).

class BasicLevel:
    """Certificate 23's basic level: base percepts with codes of 3 ones in
    ``width`` and nominated conjunction rows (the greatest of their members'
    codes and a small keyed draw, as ``BasicPercepts.code``).

    ``two_lane``: a row's against entries are the rows that hold it as a
    strict whole (the conjunctions it is a member of), joined by the union;
    its code's against lane is what those wholes add to it. A reading's
    combined vector takes each row's for code at its for value and its
    against code at its for value on the against lane, and the reverse at
    its against value."""

    def __init__(self, supports, conjunctions, *, width=24, two_lane=False, increment=.1, own=False):
        n = len(supports)
        width = max(width, 1 + max(max(s) for s in supports) + (3 * len(conjunctions) if own else 0))
        codes = torch.zeros(n + len(conjunctions), width, dtype=torch.float64)
        for row, support in enumerate(supports):
            codes[row, list(support)] = 1.
        free = 1 + max(max(s) for s in supports)
        for i, members in enumerate(conjunctions):
            if own:
                # ``own``: a conjunction has 3 coordinates of its own, as a
                # part has; what it adds to a member is those alone.
                codes[n + i] = codes[list(members)].amax(0)
                codes[n + i, free + 3 * i: free + 3 * i + 3] = 1.
                continue
            draw = torch.rand(width, generator=torch.Generator().manual_seed(n + i), dtype=torch.float64)
            codes[n + i] = (codes[list(members)].amax(0) + increment * draw).clamp(max=1.)
        self.rows = [()] * n + [tuple(c) for c in conjunctions]
        self.codes, self.two_lane = codes, two_lane
        self.holders = {r: [h for h, members in enumerate(self.rows) if r in members] for r in range(len(self.rows))}
        against = torch.zeros_like(codes)
        if two_lane:
            for r, hs in self.holders.items():
                for h in hs:
                    added = codes[h] - codes[r]
                    if own:
                        # Only the whole's own coordinates, not its other
                        # members'.
                        added = codes[h] - codes[list(self.rows[h])].amax(0)
                    against[r] = torch.maximum(against[r], added.clamp(min=0))
        self.against = against

    def lanes(self, present):
        """Each row's (for, against) on a reading of the base rows
        ``present``; a conjunction is present when all its members are."""
        on = [float(r in present) if not members else float(all(m in present for m in members))
              for r, members in enumerate(self.rows)]
        result = []
        for r in range(len(self.rows)):
            f = on[r]
            a = f * float(any(on[h] for h in self.holders[r])) if self.two_lane else 0.
            result.append((f, a))
        return torch.tensor(result, dtype=torch.float64)

    def combined(self, present):
        b = self.lanes(present)
        gaps_for = torch.log((1 - self.codes).clamp_min(1e-12))
        gaps_against = torch.log((1 - self.against).clamp_min(1e-12))
        x_for = 1 - torch.exp(b[:, :1].T @ gaps_for + b[:, 1:].T @ gaps_against)[0]
        x_against = 1 - torch.exp(b[:, :1].T @ gaps_against + b[:, 1:].T @ gaps_for)[0]
        return x_for - x_against

    def column(self, members, signs=None):
        """Rule 8b on the two lanes: each member's for less against code over
        its squared length, summed (a Not member subtracted)."""
        signs = [1.] * len(members) if signs is None else signs
        u = torch.zeros(self.codes.shape[1], dtype=torch.float64)
        for m, s in zip(members, signs):
            v = self.codes[m] - self.against[m]
            u += s * v / v.square().sum()
        return u

    def teach(self, cases, *, nonnegative, steps=2000, lr=.05):
        x = torch.stack([self.combined(set(p)) for p in cases])
        target = torch.tensor(list(cases.values()), dtype=torch.float64)
        u = torch.zeros(self.codes.shape[1], dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.Adam([u], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            ((torch.tanh(x @ u) - target) ** 2).mean().backward()
            optimizer.step()
            if nonnegative:
                with torch.no_grad():
                    u.clamp_(min=0)
        return u.detach()

    def read(self, u, present):
        return float(torch.tanh(self.combined(set(present)) @ u))


# ---------------------------------------------------------------------------
# The wholes half on the basis of the part catalog (§19.73).

def catalog_basis(coder, second=False):
    """Each catalog part's coordinates in the wholes half: the parts half's
    own (``s``) or the same catalog drawn a second time (``i``)."""
    atoms = set()
    for w in coder.words:
        atoms |= coder.parts[w]
    if not second:
        return {a: coder.atom_mask(a) for a in atoms}
    return {a: mask_of(coords(b'\x00whole-basis/' + a.encode(), coder.wholes_n, coder.bits.get(a, ONES)))
            for a in atoms}


def class_code(basis, members):
    """A whole's code: the OR of the codes of the catalog parts that
    belong to it (every character of the part in ``members``, bytes)."""
    m = 0
    for atom, mask in basis.items():
        if set(atom.encode()) <= members:
            m |= mask
    return m


def catalog_bound(coder, form, basis, *, tight, extra=None, classes=None):
    """A form's upper bound in the wholes half: the AND of its wholes'
    codes (the top, all ones, and the character classes it lies in; with
    ``tight`` the form itself, the OR of its own parts)."""
    full = (1 << coder.wholes_n) - 1
    bound = full
    classes = {} if classes is None else classes
    wholes = dict(coder.wholes, **(extra or {}))
    held = set(form.encode())
    for name, members in wholes.items():
        if held <= members:
            if name not in classes:
                classes[name] = class_code(basis, members)
            bound &= classes[name]
    if tight:
        own = 0
        for atom in coder.parts[form]:
            own |= basis[atom]
        bound &= own
    return bound


def catalog_codes(coder, *, second, tight, extra=None):
    """Every form's two halves: the parts half (for), and the wholes half
    read on its complement, the coordinates outside the upper bound, on
    the against lane. Returns (codes for less against, parts masks,
    bounds)."""
    n, m = coder.parts_n, coder.wholes_n
    basis = catalog_basis(coder, second)
    classes = {}
    parts = [coder.part_mask(w) for w in coder.words]
    bounds = [catalog_bound(coder, w, basis, tight=tight, extra=extra, classes=classes) for w in coder.words]
    codes = np.zeros((len(coder.words), n + m))
    for row, (p, b) in enumerate(zip(parts, bounds)):
        codes[row, :n] = bits(p, n)
        codes[row, n:] = -(1 - bits(b, m))
    return codes, parts, bounds, basis


# ---------------------------------------------------------------------------
# No code inside another by accident (§19.86).

def run_index(words):
    """For each form, the forms of ``words`` it holds as a run (its form
    parts) and the forms that hold it (its form wholes), by run alone."""
    stored = set(words)
    inside = {w: set() for w in words}
    holders = {w: set() for w in words}
    for w in words:
        for size in range(1, len(w)):
            for start in range(len(w) - size + 1):
                run = w[start:start + size]
                if run in stored:
                    inside[w].add(run)
                    holders[run].add(w)
    return inside, holders


class LinkedCoder:
    """The source's present coding (round 3c), at ``n`` + ``n``: a form's
    parts half is the OR of its parts' codes; its wholes half its own
    tightest whole, the OR of its parts' catalog codes (the catalog drawn a
    second time); two forms with one code are told apart by rule 43's next
    part, then by more bits on an added part (``WordIdentity._settle``,
    ``_separate``). A code is one integer of ``2n`` bits, the parts half
    low.

    With ``repair`` (§19.86), as each form is admitted every stored form
    whose code lies inside its code, or whose code its code lies inside,
    with neither a run of the other, is an accident; the form that lies
    inside gets coordinates that the other lacks, by becoming a part of the
    catalog itself, carried by every stored form that holds it as a run:
    ``a`` drawn as a part's are (both halves, more bits until it is out);
    ``b`` chosen among the parts-half coordinates the other leaves at zero;
    ``c`` chosen among the zeros of either half."""

    CATALOG = b'\x00whole-basis/'

    def __init__(self, n=64, *, repair=None, ones=ONES, cap=None):
        self.n, self.ones, self.repair = int(n), int(ones), repair
        self.width = 2 * self.n
        self.limbs = (self.width + 63) // 64
        self.atoms = {}            # atom value -> mask
        self.bits = {}             # atom value -> ones, where raised
        self.added = {}            # run -> why (('next', l, r), ('form', inner, outer))
        self.order, self.words, self.parts = {}, [], {}
        self.forms, self.keys = {}, {}
        self.codes = np.zeros((0, self.limbs), dtype=np.uint64)
        self.inside, self.holders = {}, {}
        self.stats = Counter()
        self.failures = []
        self.changed_by = []       # per admission: earlier codes changed
        self.timings = []
        self.cap = cap
        self.usage = np.zeros(self.width, dtype=np.int64)
        self.given_forms = set()

    # Codes -------------------------------------------------------------

    def drawn(self, value, k):
        parts = mask_of(bit_stream(value.encode(), self.n)[:k])
        wholes = mask_of(bit_stream(self.CATALOG + value.encode(), self.n)[:k])
        return parts | (wholes << self.n)

    def atom_mask(self, value, k=None):
        if k is None and value in self.atoms:
            return self.atoms[value]
        return self.drawn(value, self.bits.get(value, self.ones) if k is None else k)

    def code(self, form, override=None):
        m = 0
        for atom in self.parts[form]:
            m |= self.atom_mask(atom, override[1] if override and atom == override[0] else None)
        return m

    def limbs_of(self, mask):
        return np.array([(mask >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(self.limbs)], dtype=np.uint64)

    def _write(self, form):
        row = self.order[form]
        self.codes[row] = self.limbs_of(self.code(form))

    # The source's admission --------------------------------------------

    def atoms_of(self, form):
        found = set(dict.fromkeys(form))
        for size in range(2, len(form) + 1):
            for start in range(len(form) - size + 1):
                run = form[start:start + size]
                if run in self.added:
                    found.add(run)
        return frozenset(found)

    def _holders_of_run(self, run):
        if run in self.holders:
            return [w for w in self.holders[run] if w in self.order] + ([run] if run in self.order else [])
        return [w for w in self.words if run in w]

    def _add_part(self, run, why, mask=None):
        self.added[run] = why
        if mask is not None:
            self.atoms[run] = mask
        else:
            self.atoms[run] = self.drawn(run, self.bits.get(run, self.ones))
        holders = self._holders_of_run(run)
        for w in holders:
            self.parts[w] = self.parts[w] | {run}
        return holders

    def _next_part(self, left, right):
        known = set(self.atoms)
        for size in range(1, max(len(left), len(right)) + 1):
            found = (runs(left, size) ^ runs(right, size)) - known
            if found:
                return min(found)
        return None

    def _separate(self, left, right):
        changed = []
        alike = lambda: self.code(left) == self.code(right)
        while alike():
            run = self._next_part(left, right)
            if run is None:
                break
            changed += self._add_part(run, ('next', left, right))
            self.stats['next parts'] += 1
        if not alike():
            return changed
        held = Counter(a for w in self.words for a in self.parts[w])
        candidates = sorted((a for a in self.parts[left] ^ self.parts[right] if a in self.added),
                            key=lambda a: (held[a], -len(a), a))
        for atom in candidates:
            for k in range(self.bits.get(atom, self.ones) + 1, self.n + 1):
                if self.code(left, (atom, k)) != self.code(right, (atom, k)):
                    self.bits[atom] = k
                    self.atoms[atom] = self.drawn(atom, k)
                    self.stats['bits raised'] += 1
                    return changed + [w for w in self.words if atom in self.parts[w]]
        return None

    def _settle(self, pending):
        affected, pending = set(pending), list(pending)
        while pending:
            word = pending.pop()
            old = self.keys.pop(word, None)
            if old is not None and self.forms.get(old) == word:
                del self.forms[old]
            key = self.code(word)
            other = self.forms.get(key)
            if other is not None and other != word:
                touched = self._separate(other, word)
                if touched is None:
                    raise RuntimeError(f'no part tells {other!r} from {word!r}')
                old = self.keys.pop(other, None)
                if old is not None and self.forms.get(old) == other:
                    del self.forms[old]
                again = sorted(set(touched) | {word, other}, key=self.order.__getitem__)
                affected.update(again)
                pending.extend(again)
            else:
                self.forms[key] = word
                self.keys[word] = key
        return affected

    def _own_part(self, form):
        """The stored forms that take ``form``'s own part as it is admitted:
        none here (the source before rule 43 was amended)."""
        return []

    def admit(self, form, index=None):
        if form in self.order:
            return
        start = time.perf_counter()
        self.order[form] = len(self.words)
        self.words.append(form)
        if len(self.words) > len(self.codes):
            self.codes = np.concatenate((self.codes, np.zeros((max(1024, len(self.codes)), self.limbs), dtype=np.uint64)))
        carriers = self._own_part(form)
        self.parts[form] = self.atoms_of(form)
        for atom in self.parts[form]:
            if atom not in self.atoms:
                self.atoms[atom] = self.drawn(atom, self.ones)
        affected = self._settle(carriers + [form])
        for w in affected | {form}:
            self._write(w)
        if self.repair:
            affected |= self._repair(set(affected) | {form})
        self.changed_by.append(len(affected - {form}))
        self.timings.append(time.perf_counter() - start)

    def admit_all(self, forms, *, index=None, progress=None, budget=None):
        """Admit ``forms`` in order; ``progress`` is called every 1,000
        forms; with ``budget`` (seconds) admission stops when it is spent and
        ``self.stopped`` says where."""
        if index is None:
            index = run_index(forms)
        self.inside, self.holders = index
        self.codes = np.zeros((len(forms), self.limbs), dtype=np.uint64)
        self.stopped = None
        start = time.perf_counter()
        for k, f in enumerate(forms):
            self.admit(f)
            if progress is not None and (k + 1) % 1000 == 0:
                progress(self)
            if budget is not None and time.perf_counter() - start > budget:
                self.stopped = k + 1
                break
        return self

    # §19.86: the repair ------------------------------------------------

    def is_run(self, inner, outer):
        return inner in outer and inner != outer

    def contained(self, mask, live=None):
        """Stored rows whose codes lie inside ``mask``, and those ``mask``
        lies inside; over the first ``live`` rows (all rows by default)."""
        c = self.limbs_of(mask)
        codes = self.codes if live is None else self.codes[:live]
        inner = ((codes & ~c) == 0).all(1)
        outer = ((c & ~codes) == 0).all(1)
        return inner, outer

    def accidents_of(self, x):
        """The accidents of ``x`` among the stored forms: pairs (inner,
        outer)."""
        c = self.code(x)
        live = len(self.words)
        self.codes[self.order[x]] = self.limbs_of(c)
        inner, outer = self.contained(c, live)
        found = []
        for j in np.flatnonzero(inner[:live]):
            y = self.words[j]
            if y != x and not self.is_run(y, x):
                found.append((y, x))
        for j in np.flatnonzero(outer[:live]):
            y = self.words[j]
            if y != x and not self.is_run(x, y):
                found.append((x, y))
        return found

    def _repair(self, start):
        changed = set()
        queue = sorted(start, key=self.order.__getitem__)
        seen_steps = 0
        while queue:
            x = queue.pop()
            for w in [x]:
                self._write(w)
            found = self.accidents_of(x)
            if not found:
                continue
            by_inner = {}
            for inner, outer in found:
                by_inner.setdefault(inner, []).append(outer)
            for inner, outers in by_inner.items():
                touched = self._separate_inner(inner, outers)
                if touched is None:
                    continue
                for w in touched:
                    self._write(w)
                changed |= set(touched)
                queue.extend(sorted(set(touched) - set(queue), key=self.order.__getitem__))
            seen_steps += 1
            if self.cap and seen_steps > self.cap:
                self.stats['repair loops cut'] += 1
                break
        return changed

    def _separate_inner(self, inner, outers):
        """Give ``inner`` coordinates that each of ``outers`` lacks, by
        making it (or its coordinates, if it is one already) a part of the
        catalog carried by every stored form that holds it as a run."""
        full = (1 << self.width) - 1
        allowed = (1 << self.n) - 1 if self.repair in ('b', 'bh') else full
        own = self.code(inner)
        holder_codes = None
        if self.repair in ('bh', 'ch'):
            # ``bh``/``ch``: the coordinate that adds the fewest new ones to
            # the form's holders, which carry it.
            holder_codes = [self.code(h) for h in self._holders_of_run(inner) if h != inner]
        if self.repair == 'a':
            k = self.bits.get(inner, self.ones) if inner in self.added else self.ones - 1
            while True:
                k += 1
                if k > self.n:
                    self.failures.append((inner, tuple(outers)))
                    return None
                mask = self.drawn(inner, k)
                if all(((own | mask) & ~self.code(o)) for o in outers):
                    break
            self.bits[inner] = k
            given = mask
        else:
            given = 0
            pending = list(outers)
            while pending:
                best, cover = None, []
                zeros = {}
                for o in pending:
                    z = ~self.code(o) & allowed & ~own & ~given
                    zeros[o] = z
                candidates = 0
                for z in zeros.values():
                    candidates |= z
                if not candidates:
                    self.failures.append((inner, tuple(pending)))
                    return None
                best_score = None
                for i in range(self.width):
                    if not (candidates >> i) & 1:
                        continue
                    covers = [o for o, z in zeros.items() if (z >> i) & 1]
                    added = 0 if holder_codes is None else sum(1 for h in holder_codes if not (h >> i) & 1)
                    score = (len(covers), -added, -int(self.usage[i]))
                    if best_score is None or score > best_score:
                        best, best_score, cover = i, score, covers
                given |= 1 << best
                pending = [o for o in pending if o not in cover]
        self.stats['coordinates given'] += popcount(given & ~self.atom_mask(inner) if inner in self.atoms else given)
        self.stats['repairs'] += 1
        for i in range(self.width):
            if (given >> i) & 1:
                self.usage[i] += 1
        if inner in self.atoms:
            self.atoms[inner] |= given
            holders = self._holders_of_run(inner)
        else:
            holders = self._add_part(inner, ('form', inner, tuple(outers)), mask=given)
        self.given_forms.add(inner)
        return sorted(set(holders) | {inner}, key=self.order.__getitem__)

    # Reports -----------------------------------------------------------

    def all_accidents(self):
        """Every ordered pair of stored forms whose codes lie one inside the
        other with neither a run of the other; and the true containments
        that fail (a run whose code is not inside its holder's)."""
        accidents = true_out = 0
        live = len(self.words)
        for i, x in enumerate(self.words):
            c = self.codes[i]
            inner = ((self.codes[:live] & ~c) == 0).all(1)
            inner[i] = False
            for j in np.flatnonzero(inner):
                if not self.is_run(self.words[j], x):
                    accidents += 1
            for part in self.inside.get(x, ()):
                if part in self.order and np.any(self.codes[self.order[part]] & ~c):
                    true_out += 1
        return accidents, true_out


class SourceCoder(LinkedCoder):
    """The stand-in brought to the source's repair (rule 61, as round 3d
    built it in ``WordIdentity._separate_inner``): ``ch``, with three
    things the trial's ``ch`` lacks, and rule 43 as amended (every stored
    form a part of its own).

    - Its keys are kept fresh after a repair, so rule 56's next part sees
      the codes as they are.
    - Among the coordinates that cover the most outers, one off the inner
      form's against lane (the parts-half coordinates that its nearest
      stored holders add) is taken where there is one.
    - In the wholes half, a coordinate inside the AND of the codes of the
      declared classes that hold on every byte of the inner form is taken
      where any covers (the source's ``_kept``)."""

    def __init__(self, n=64, **kwargs):
        super().__init__(n, repair='ch', **kwargs)
        self._classes = None

    def _write(self, form):
        super()._write(form)
        new, old = self.code(form), self.keys.get(form)
        if old is not None and old != new:
            if self.forms.get(old) == form:
                del self.forms[old]
            self.keys[form] = new
            self.forms[new] = form

    def _own_part(self, form):
        # Rule 43 as amended: a stored form of several characters is a part
        # of its own, carried by every stored form that holds it as a run.
        if len(form) < 2 or form in self.added:
            return []
        self.added[form] = ('own', form)
        carriers = sorted((h for h in self._holders_of_run(form) if h != form), key=self.order.__getitem__)
        for h in carriers:
            self.parts[h] = self.parts[h] | {form}
        return carriers

    def _opposed(self, inner):
        holders = [h for h in self._holders_of_run(inner) if h != inner]
        if not holders:
            return 0
        least = min(map(len, holders))
        # A holder's part that holds the inner form itself (its own part) is
        # found only where the characters it adds are: it opposes nothing.
        added = {a for a in set().union(*(self.parts[h] for h in holders if len(h) == least)) - self.parts[inner]
                 if inner not in a}
        mask = 0
        for atom in added:
            mask |= self.atom_mask(atom)
        return mask & ((1 << self.n) - 1) & ~self.code(inner)

    def _kept(self, inner):
        if self._classes is None:
            self._classes = []
            for members in declared_wholes().values():
                m = 0
                for b in members:
                    m |= mask_of(bit_stream(self.CATALOG + bytes([b]), self.n)[:self.ones])
                self._classes.append((members, m << self.n))
        bound = ((1 << self.n) - 1) << self.n
        held = set(inner.encode())
        for members, m in self._classes:
            if held <= members:
                bound &= m
        return ((1 << self.n) - 1) | bound

    def _separate_inner(self, inner, outers):
        own = self.code(inner)
        holder_codes = [self.code(h) for h in self._holders_of_run(inner) if h != inner]
        opposed = self._opposed(inner)
        kept = self._kept(inner)
        given, pending = 0, list(outers)
        while pending:
            zeros = {o: ~self.code(o) & ((1 << self.width) - 1) & ~own & ~given for o in pending}
            candidates = 0
            for z in zeros.values():
                candidates |= z
            if not candidates:
                self.failures.append((inner, tuple(pending)))
                return None
            inside = any(z & kept for z in zeros.values())
            scored = []
            for i in range(self.width):
                if (candidates >> i) & 1:
                    covers = [o for o, z in zeros.items() if (z >> i) & 1]
                    counted = [o for o in covers if (kept >> i) & 1] if inside else covers
                    added = sum(1 for h in holder_codes if not (h >> i) & 1)
                    scored.append(((len(counted), -((opposed >> i) & 1), -added, -int(self.usage[i]), -i), i, covers))
            _score, best, cover = max(scored)
            given |= 1 << best
            pending = [o for o in pending if o not in cover]
        self.stats['repairs'] += 1
        for i in range(self.width):
            if (given >> i) & 1:
                self.usage[i] += 1
        if inner in self.atoms:
            self.atoms[inner] |= given
            holders = self._holders_of_run(inner)
        else:
            holders = self._add_part(inner, ('form', inner, tuple(outers)), mask=given)
        self.given_forms.add(inner)
        return sorted(set(holders) | {inner}, key=self.order.__getitem__)


def ones_in_halves(coder):
    p = [popcount(coder.code(w) & ((1 << coder.n) - 1)) for w in coder.words]
    h = [popcount(coder.code(w) >> coder.n) for w in coder.words]
    return dict(parts=(min(p), round(float(np.mean(p)), 1), max(p)),
                wholes=(min(h), round(float(np.mean(h)), 1), max(h)),
                full_parts_half=sum(x == coder.n for x in p), full_wholes_half=sum(x == coder.n for x in h))


def sample_accidents(coder, every=40):
    """The sample's count (every 40th form against all): codes inside
    another's with the one not a run of the other, and inside truly."""
    live = len(coder.words)
    accident = true = 0
    for i in range(0, live, every):
        c = coder.codes[i]
        outer = ((c & ~coder.codes[:live]) == 0).all(1)
        outer[i] = False
        for j in np.flatnonzero(outer):
            if coder.is_run(coder.words[i], coder.words[j]):
                true += 1
            else:
                accident += 1
    return dict(true=true, accident=accident)


def admission_times(coder, marks=(1000, 5000, 19821), window=100):
    """Mean seconds to admit one form near each count stored, and the
    greatest."""
    t = np.array(coder.timings)
    result = {}
    for mark in marks:
        lo, hi = max(0, min(mark, len(t)) - window), min(mark, len(t))
        if hi > lo:
            result[mark] = (round(float(t[lo:hi].mean() * 1000), 2), round(float(t[lo:hi].max() * 1000), 1))
    return result
