"""The felt world: sentences read beside small images of pixel values.

A recipe (data/felt/recipes/category-NN-<name>.json) holds one category's
frames: each frame's text with its slots, the predicate its meaning is, and
what its images must show; data/felt/recipes/cast.json holds the nine values,
the two kinds and every form of their words. This module draws worlds,
evaluates the predicates of the design on their images, writes documents from
the recipes, verifies frozen rows and loads them. It holds no word of any
language: a word reaches a sentence only from a recipe or the cast. Item 1.6
plan, section 14.
"""
import argparse
import hashlib
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path

from math_chain_corpus import HELD_OUT_PAIRS

ROOT = Path(__file__).resolve().parents[1]
FELT = ROOT / 'data' / 'felt'
SIDE = 8
VALUES = tuple(range(1, 10))
STEP = 25
SPREAD = 3
SPLITS = ('train', 'validation', 'test', 'generalization', 'witness', 'inference', 'contrast')
UNITS = ((0, -1), (0, 1), (-1, 0), (1, 0))
SLOT = re.compile(r'\{(\w+)\}')
ATTEMPTS = 400          # worlds drawn for one document before its frames are given up
NEAR, FAR = 2, 5        # section 14.2: near is at most 2, far at least 5
BIG = 4                 # big is 4 or more places, small is 1
ORDERS = ('row_before', 'column_before', 'level_before', 'bigger', 'precedes')


# -- values, bands and what is reserved (section 14.1, 14.5) ------------------

def band(d):
    return range(STEP * d - SPREAD, STEP * d + SPREAD + 1)


def reserved_step(d):
    return STEP * d + 1 if d % 2 else STEP * d - 1


def reserved_places(d):
    return frozenset((x, y) for x in range(SIDE) for y in range(SIDE) if (x + y + d) % 8 == 0)


RESERVED = {d: reserved_places(d) for d in VALUES}
FREE_STEPS = {d: tuple(s for s in band(d) if s != reserved_step(d)) for d in VALUES}


def value_of(steps):
    return (steps + SPREAD) // STEP


def inside(p):
    return 0 <= p[0] < SIDE and 0 <= p[1] < SIDE


def add(p, u, k=1):
    return (p[0] + u[0] * k, p[1] + u[1] * k)


# -- a document's images as places and steps -----------------------------------

def frame_of(scene):
    """A scene ([[x, y, z, steps], ...] or None) as {value: (places, steps)}."""
    if scene is None:
        return None
    things = {}
    for x, y, _z, steps in scene:
        places, levels = things.setdefault(value_of(steps), (set(), []))
        places.add((x, y))
        levels.append(steps)
    return {v: (frozenset(p), tuple(sorted(s))) for v, (p, s) in things.items()}


def scene_of(arrangement, rng, *, reserved=False):
    """{value: places} as a scene, each place's steps drawn afresh."""
    if arrangement is None:
        return None
    rows = []
    for v in sorted(arrangement):
        for x, y in sorted(arrangement[v]):
            rows.append([x, y, 0, reserved_step(v) if reserved else rng.choice(FREE_STEPS[v])])
    rows.sort()
    return rows


def distance(a, b):
    return min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p in a for q in b)


def touching(a, b):
    return any(abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1 for p in a for q in b)


def shift(a, b):
    dx = min(b)[0] - min(a)[0]
    dy = min(b)[1] - min(a)[1]
    return (dx, dy) if frozenset(add(p, (dx, dy)) for p in a) == b else None


@dataclass(frozen=True)
class Event:
    first: int      # its first image
    last: int       # its last image
    n: int | None   # places at every step; None if the steps differ
    unit: tuple | None


class View:
    """The images of one document, with the events of each thing."""

    def __init__(self, frames, cast):
        self.frames = frames
        self.cast = cast
        self.count = len(frames)
        self.values = sorted({v for f in frames if f for v in f})
        self.events = {v: self._events(v) for v in self.values}

    def places(self, v, t):
        f = self.frames[t]
        if f is None or v not in f:
            return None
        return f[v][0]

    def steps(self, v, t):
        return self.frames[t][v][1]

    def displaced(self, v, t):
        a, b = self.places(v, t), self.places(v, t + 1)
        return a is not None and b is not None and a != b

    def _events(self, v):
        runs, t = [], 0
        while t < self.count - 1:
            if self.displaced(v, t):
                s = t
                while t < self.count - 1 and self.displaced(v, t):
                    t += 1
                vectors = {shift(self.places(v, k), self.places(v, k + 1)) for k in range(s, t)}
                n = unit = None
                if len(vectors) == 1:
                    vec = next(iter(vectors))
                    if vec is not None and (vec[0] == 0) != (vec[1] == 0) and abs(vec[0] + vec[1]) in (1, 2):
                        n = abs(vec[0] + vec[1])
                        unit = (vec[0] // n, vec[1] // n)
                runs.append(Event(s, t, n, unit))
            else:
                t += 1
        return runs

    def at_rest(self, v, t):
        sides = [self.displaced(v, k) for k in (t - 1, t) if 0 <= k < self.count - 1]
        if not sides:
            return None
        return not any(sides)

    def done(self, v, t):
        """Events of v complete by image t; None while one is under way."""
        if any(e.first < t < e.last for e in self.events.get(v, ())):
            return None
        return [e for e in self.events.get(v, ()) if e.last <= t]

    def members(self, k, t):
        f = self.frames[t]
        return [v for v in sorted(f) if v in self.cast.members[k]] if f else []

    def unmoved_over(self, v, e):
        return all(self.places(v, k) == self.places(v, e.first) for k in range(e.first, e.last + 1))

    def gaps(self, a, b, e):
        return [distance(self.places(a, k), self.places(b, k)) for k in range(e.first, e.last + 1)]

    def approach(self, a, b, e):
        g = self.gaps(a, b, e)
        return self.unmoved_over(b, e) and all(x > y for x, y in zip(g, g[1:]))

    def recede(self, a, b, e):
        g = self.gaps(a, b, e)
        return self.unmoved_over(b, e) and all(x < y for x, y in zip(g, g[1:]))


class Cast:
    def __init__(self, cast):
        self.data = cast
        self.word = {entry['value']: entry['word'] for entry in cast['values']}
        self.kinds = {entry['key']: entry for entry in cast['kinds']}
        self.members = {k: frozenset(entry['members']) for k, entry in self.kinds.items()}
        self.kind_keys = sorted(self.kinds)

    def kind_of(self, d):
        return next(k for k in self.kind_keys if d in self.members[k])


# -- the predicates of section 14.2 and of the conditions under 14.4 ----------
#
# Each takes the view, the image of "now" and its arguments, and returns True,
# False, or None where the frame's presupposition fails (a graded word between
# its bounds, an answer that is not unique, a thing absent that must be there).

def both(w, t, *vs):
    return all(w.places(v, t) is not None for v in vs)


def three(*values):
    if any(v is None for v in values):
        return None
    return all(values)


def p_word(w, t, a):
    return True


def p_this_is(w, t, a):
    f = w.frames[t]
    return None if f is None else sorted(f) == [a]


def p_this_one_of(w, t, a, b):
    f = w.frames[t]
    return None if f is None else sorted(f) in ([a], [b])


def p_present(w, t, a):
    f = w.frames[t]
    return None if f is None else a in f


def p_absent(w, t, a):
    f = w.frames[t]
    return None if f is None else a not in f


def p_both_present(w, t, a, b):
    f = w.frames[t]
    return None if f is None else (a in f and b in f)


def p_one_present(w, t, a, b):
    f = w.frames[t]
    return None if f is None else ((a in f) != (b in f))


def p_member(w, t, a, k):
    return a in w.cast.members[k]


def p_not_member(w, t, a, k):
    return a not in w.cast.members[k]


def p_kinds_both(w, t, a, k, b, l):
    return a in w.cast.members[k] and b in w.cast.members[l]


def p_kind_members(w, t, k):
    return True


def p_level_before(w, t, a, b):
    if w.frames[t] is None:
        return said_level(a, b)
    if not both(w, t, a, b):
        return None
    return max(w.steps(a, t)) < min(w.steps(b, t))


def p_not_level_before(w, t, a, b):
    r = p_level_before(w, t, a, b)
    return None if r is None else not r


def said_level(a, b):
    return a < b


def p_row_before(w, t, a, b):
    if not both(w, t, a, b):
        return None
    pa, pb = w.places(a, t), w.places(b, t)
    return (max(y for _, y in pa) < min(y for _, y in pb)
            and bool({x for x, _ in pa} & {x for x, _ in pb}))


def p_column_before(w, t, a, b):
    if not both(w, t, a, b):
        return None
    pa, pb = w.places(a, t), w.places(b, t)
    return (max(x for x, _ in pa) < min(x for x, _ in pb)
            and bool({y for _, y in pa} & {y for _, y in pb}))


def p_adjacent(w, t, a, b):
    if not both(w, t, a, b):
        return None
    return touching(w.places(a, t), w.places(b, t))


def p_on_top(w, t, a, b):
    return three(p_row_before(w, t, a, b), p_adjacent(w, t, a, b))


def p_next_to(w, t, a, b):
    if not both(w, t, a, b):
        return None
    return any(p[1] == q[1] and abs(p[0] - q[0]) == 1 for p in w.places(a, t) for q in w.places(b, t))


def p_near(w, t, a, b):
    if not both(w, t, a, b):
        return None
    d = distance(w.places(a, t), w.places(b, t))
    return True if d <= NEAR else False if d >= FAR else None


def p_far(w, t, a, b):
    r = p_near(w, t, a, b)
    return None if r is None else not r


def p_between(w, t, a, b, c):
    if not both(w, t, a, b, c):
        return None
    pa, pb, pc = w.places(a, t), w.places(b, t), w.places(c, t)
    for axis in (0, 1):
        other = 1 - axis
        if len({p[other] for p in pa | pb | pc}) == 1:
            lo, hi = sorted((pb, pc), key=lambda s: min(p[axis] for p in s))
            if (max(p[axis] for p in lo) < min(p[axis] for p in pa)
                    and max(p[axis] for p in pa) < min(p[axis] for p in hi)):
                return True
    return False


def size(w, t, a):
    return len(w.places(a, t))


def p_big(w, t, a):
    if not both(w, t, a):
        return None
    s = size(w, t, a)
    return True if s >= BIG else False if s == 1 else None


def p_small(w, t, a):
    r = p_big(w, t, a)
    return None if r is None else not r


def p_bigger(w, t, a, b):
    if not both(w, t, a, b):
        return None
    sa, sb = size(w, t, a), size(w, t, b)
    return None if sa == sb else sa > sb


def p_count(w, t):
    f = w.frames[t]
    return None if f is None else len(f)


def p_moved(w, t, a):
    if not both(w, t, a):
        return None
    done = w.done(a, t)
    return None if done is None else bool(done)


def p_did_not_move(w, t, a):
    if not both(w, t, a):
        return None
    if any(w.displaced(a, k) for k in range(t)):
        return False
    others = [v for v in w.values if v != a and w.done(v, t)]
    return True if others else None


def p_stopped(w, t, a):
    if not both(w, t, a):
        return None
    done = w.done(a, t)
    if done is None:
        return None
    if not done:
        return False if w.at_rest(a, t) else None
    return True if w.at_rest(a, t) else None


def p_moving(w, t, a):
    if not both(w, t, a):
        return None
    before = w.displaced(a, t - 1) if t >= 1 else None
    after = w.displaced(a, t) if t < w.count - 1 else None
    if before and after:
        return True
    if before is False and after is False:
        return False
    return None


def p_was_moving(w, t, a):
    if not both(w, t, a) or not w.at_rest(a, t):
        return None
    return any(e.last <= t and e.last - e.first >= 2 for e in w.events.get(a, ()))


def p_will_move(w, t, a):
    if not both(w, t, a) or not w.at_rest(a, t):
        return None
    if any(w.displaced(a, k) for k in range(t)):
        return None
    return any(w.displaced(a, k) for k in range(t, w.count - 1))


def regular(events):
    return all(e.n is not None for e in events)


def p_moved_speed(w, t, a, n):
    moved = p_moved(w, t, a)
    if not moved:
        return moved
    done = w.done(a, t)
    if not regular(done):
        return None
    speeds = {e.n for e in done}
    return None if len(speeds) > 1 else speeds == {n}


def p_moved_dir(w, t, a, unit):
    moved = p_moved(w, t, a)
    if not moved:
        return None if moved is False else moved
    done = w.done(a, t)
    if not regular(done):
        return None
    units = {e.unit for e in done}
    return None if len(units) > 1 else units == {tuple(unit)}


def p_reached(w, t, a, b):
    if not both(w, t, a, b):
        return None
    done = w.done(a, t)
    if done is None:
        return None
    for e in done:
        if (w.approach(a, b, e) and touching(w.places(a, e.last), w.places(b, e.last))
                and not any(touching(w.places(a, k), w.places(b, k)) for k in range(e.first, e.last))):
            return True
    return False


def p_left_from(w, t, a, b):
    if not both(w, t, a, b):
        return None
    done = w.done(a, t)
    if done is None:
        return None
    return any(touching(w.places(a, e.first), w.places(b, e.first)) and w.recede(a, b, e) for e in done)


def completed_rest(w, t, a, b):
    if not both(w, t, a, b):
        return None
    done = w.done(a, t)
    if done is None or not w.at_rest(a, t):
        return None
    return done


def p_moved_toward(w, t, a, b):
    done = completed_rest(w, t, a, b)
    return done if done is None else any(w.approach(a, b, e) for e in done)


def p_moved_toward_speed(w, t, a, b, n):
    done = completed_rest(w, t, a, b)
    return done if done is None else any(w.approach(a, b, e) and e.n == n for e in done)


def p_moved_away(w, t, a, b):
    done = completed_rest(w, t, a, b)
    return done if done is None else any(w.recede(a, b, e) for e in done)


def p_moving_toward(w, t, a, b):
    if not both(w, t, a, b):
        return None
    if w.at_rest(a, t):
        return False
    within = [e for e in w.events.get(a, ()) if e.first < t < e.last]
    return any(w.approach(a, b, e) for e in within) if within else None


SIDES = {(1, 0): ('column_before', False), (-1, 0): ('column_before', True),
         (0, 1): ('row_before', False), (0, -1): ('row_before', True)}


def p_moving_toward_from(w, t, a, b, unit):
    toward = p_moving_toward(w, t, a, b)
    if not toward:
        return toward
    order, swap = SIDES[tuple(unit)]
    test = PREDICATES[order]
    e = next(e for e in w.events[a] if e.first < t < e.last)
    return all(test(w, k, *((b, a) if swap else (a, b))) for k in range(e.first, e.last + 1))


def p_toward_quickly_beside_slow(w, t, a, b):
    quick = p_moved_toward_speed(w, t, a, b, 2)
    if not quick:
        return quick
    return any(p_moved_toward_speed(w, t, v, b, 1) for v in w.values if v not in (a, b))


def p_was_above(w, t, a, b):
    now = p_row_before(w, t, a, b)
    if now is None:
        return None
    if now:
        return False
    return any(p_row_before(w, k, a, b) for k in range(t))


def p_precedes(w, t, a, b):
    if not both(w, t, a, b):
        return None
    da, db = w.done(a, t), w.done(b, t)
    if da is None or db is None or len(da) != 1 or len(db) != 1:
        return None
    if da[0].last < db[0].first:
        return True
    if db[0].last < da[0].first:
        return False
    return None


def p_moved_and_stopped(w, t, a, b):
    return three(p_moved(w, t, a), p_stopped(w, t, b))


def p_moved_but_not(w, t, a, b):
    return three(p_moved(w, t, a), p_did_not_move(w, t, b))


def p_moved_both(w, t, a, b):
    return three(p_moved(w, t, a), p_moved(w, t, b))


def kind_movers(w, t, k):
    m = w.members(k, t)
    if len(m) < 2:
        return None, None
    moved = [p_moved(w, t, v) for v in m]
    if any(x is None for x in moved):
        return None, None
    return m, [v for v, x in zip(m, moved) if x]


def p_kind_moved_one(w, t, k):
    m, movers = kind_movers(w, t, k)
    if m is None:
        return None
    return True if len(movers) == 1 else False if not movers else None


def p_kind_moved_all(w, t, k):
    m, movers = kind_movers(w, t, k)
    return None if m is None else len(movers) == len(m)


def p_kind_moved_both(w, t, k):
    m, movers = kind_movers(w, t, k)
    if m is None or len(m) != 2:
        return None
    return len(movers) == 2


def p_kind_moved_some(w, t, k):
    m, movers = kind_movers(w, t, k)
    return None if m is None else 2 <= len(movers) < len(m)


def p_kind_moved_none(w, t, k):
    m, movers = kind_movers(w, t, k)
    if m is None:
        return None
    if movers:
        return False
    others = [v for v in w.values if v not in m and w.places(v, t) is not None and p_moved(w, t, v)]
    return True if others else None


def p_kind_stopped_all(w, t, k):
    m = w.members(k, t)
    if len(m) < 2:
        return None
    return three(*[p_stopped(w, t, v) for v in m])


def sized(w, t, m):
    sizes = [size(w, t, v) for v in m]
    if any(1 < s < BIG for s in sizes):
        return None, None
    return [v for v, s in zip(m, sizes) if s >= BIG], [v for v, s in zip(m, sizes) if s == 1]


def p_the_small_moved(w, t, k):
    m = w.members(k, t)
    if len(m) < 2:
        return None
    big, small = sized(w, t, m)
    if small is None or len(small) != 1:
        return None
    return p_moved(w, t, small[0])


def p_the_big(w, t, a, k):
    m = w.members(k, t)
    if len(m) < 2:
        return None
    big, small = sized(w, t, m)
    if big is None or len(big) != 1:
        return None
    return a == big[0]


def p_present_member(w, t, a, k):
    f = w.frames[t]
    return None if f is None else (a in f and a in w.cast.members[k])


def clause(w, t, k, test):
    m = w.members(k, t)
    if len(m) < 2:
        return None
    holds = [test(v) for v in m]
    if any(x is None for x in holds):
        holds = [bool(x) for x in holds]
    chosen = [v for v, x in zip(m, holds) if x]
    return chosen[0] if len(chosen) == 1 else None


def p_the_k_above_moved(w, t, k, b):
    if not both(w, t, b):
        return None
    v = clause(w, t, k, lambda v: v != b and p_row_before(w, t, v, b))
    return None if v is None else p_moved(w, t, v)


def p_the_k_moved_is(w, t, k, a):
    v = clause(w, t, k, lambda v: p_moved(w, t, v))
    return None if v is None else v == a


def p_the_k_next_is(w, t, k, b, a):
    if not both(w, t, b):
        return None
    v = clause(w, t, k, lambda v: v != b and p_next_to(w, t, v, b))
    return None if v is None else v == a


def p_the_k_above(w, t, a, k, b):
    if not both(w, t, b):
        return None
    v = clause(w, t, k, lambda v: v != b and p_row_before(w, t, v, b))
    return None if v is None else v == a


def every_member(w, t, k, b, test):
    if b in w.cast.members[k] or not both(w, t, b):
        return None
    m = w.members(k, t)
    if len(m) < 2:
        return None
    return three(*[test(v) for v in m])


def p_every_above(w, t, k, b):
    return every_member(w, t, k, b, lambda v: p_row_before(w, t, v, b))


def p_every_below(w, t, k, b):
    return every_member(w, t, k, b, lambda v: p_row_before(w, t, b, v))


def p_none_next_to(w, t, k, b):
    r = every_member(w, t, k, b, lambda v: p_next_to(w, t, v, b))
    if r is None:
        return None
    return not any(p_next_to(w, t, v, b) for v in w.members(k, t))


def p_if_above_below(w, t, a, b):
    return True if both(w, t, a, b) else None


def moved_steps(w, a):
    return {k for k in range(w.count - 1) if w.displaced(a, k)}


def p_if_moves_moves(w, t, a, b):
    if w.count - 1 < 3 or not both(w, t, a, b):
        return None
    sa = moved_steps(w, a)
    if len(sa) < 2:
        return None
    return sa <= moved_steps(w, b)


def p_if_moves_still(w, t, a, b):
    if w.count - 1 < 3 or not both(w, t, a, b):
        return None
    sa, sb = moved_steps(w, a), moved_steps(w, b)
    if len(sa) < 2:
        return None
    if sa & sb:
        return False
    return True if sb else None


def p_succ(w, t, a, b):
    return b == a + 1


def p_sum(w, t, a, b, c):
    return a + b == c


# Questions whose predicate gives the answer itself.

def p_greater_of(w, t, a, b):
    r = p_level_before(w, t, a, b)
    return None if r is None else (b if r else a)


def p_bigger_of(w, t, a, b):
    r = p_bigger(w, t, a, b)
    return None if r is None else (a if r else b)


def p_slower_of(w, t, a, b):
    sa, sb = p_moved_speed(w, t, a, 1), p_moved_speed(w, t, b, 1)
    qa, qb = p_moved_speed(w, t, a, 2), p_moved_speed(w, t, b, 2)
    if sa and qb:
        return a
    if sb and qa:
        return b
    return None


def p_first_of(w, t, a, b):
    r = p_precedes(w, t, a, b)
    return None if r is None else (a if r else b)


def two_moves(w, t, a):
    done = w.done(a, t)
    if not done or len(done) != 2 or not regular(done) or done[0].n == done[1].n:
        return None
    return done


def p_faster_move(w, t, a):
    done = two_moves(w, t, a)
    return None if done is None else (0 if done[0].n > done[1].n else 1)


def p_slower_move(w, t, a):
    done = two_moves(w, t, a)
    return None if done is None else (0 if done[0].n < done[1].n else 1)


def p_kind_of(w, t, a):
    return w.cast.kind_of(a)


PREDICATES = {name[2:]: fn for name, fn in list(globals().items()) if name.startswith('p_') and callable(fn)}
PRUNE = None  # set below: predicates whose truth the slots alone decide while drawing them
SAID = {'word', 'member', 'not_member', 'kinds_both', 'kind_members', 'succ', 'sum', 'kind_of'}


PRUNE = SAID | {'level_before', 'not_level_before'}


# -- frames, slots and words ---------------------------------------------------

def literal(arg):
    return isinstance(arg, dict) and '=' in arg


def resolve(args, slots, query=None):
    out = []
    for arg in args:
        if literal(arg):
            value = arg['=']
            out.append(tuple(value) if isinstance(value, list) else value)
        elif arg == query:
            out.append('?')
        else:
            out.append(slots[arg])
    return out


def text_slots(text, recipe):
    return [recipe['forms'][name][0] for name in SLOT.findall(text)]


def named_slots(frame, recipe):
    """The slots a frame names: in its text, or, for an image shown with no text, its arguments."""
    if frame['text']:
        return text_slots(frame['text'], recipe)
    return [a for a in frame['args'] if isinstance(a, str)]


def render(text, slots, recipe, cast):
    def form(match):
        slot, field = recipe['forms'][match.group(1)]
        value = slots[slot]
        if recipe['slots'][slot] == 'digit':
            return cast.word[value]
        return cast.kinds[value][field]
    return SLOT.sub(form, text)


def answer_word(spec, value, cast):
    kind = spec['type']
    if kind == 'yes_no':
        return cast.data['truth_words'][int(value)]
    if kind == 'count':
        return ' '.join(cast.data['counting'][:value])
    if 'words' in spec:
        return spec['words'][value]
    if 'form' in spec:
        return cast.kinds[value][spec['form']]
    return cast.word[value]


def evaluate(record, view, frame):
    """The value of a meaning record on the images: a truth, or an answer."""
    t = record['entry']
    fn = PREDICATES[record['predicate']]
    args = [tuple(a) if isinstance(a, list) else a for a in record['args']]
    if '?' in args:
        i = args.index('?')
        # A question of the images is not answered by a thing its own arguments name.
        pool = VALUES if record['predicate'] in SAID else [v for v in view.values if v not in args]
        found = {}
        for v in pool:
            found[v] = fn(view, t, *args[:i], v, *args[i + 1:])
        if any(x is None for x in found.values()):
            return None
        chosen = [v for v, x in found.items() if x]
        return chosen[0] if len(chosen) == 1 else None
    return fn(view, t, *args)


def instance_cast(frame, slots, recipe, value=None):
    """The digits a frame instance names, with a wh question's answer."""
    named = [slots[s] for s in named_slots(frame, recipe) if recipe['slots'][s] == 'digit']
    spec = frame.get('answer')
    if spec and spec['type'] == 'wh' and isinstance(value, int):
        named.append(value)
    return sorted(set(named))


def held_out(recipe, index, frame, slots, digits):
    rule = recipe['held_out']
    if rule == 'pairs':
        return any(p in digits and q in digits for p, q in HELD_OUT_PAIRS)
    if rule == 'frame_digit':
        return any((index + d) % 4 == 0 for d in digits)
    if rule == 'sums':
        return frame['predicate'] == 'sum' and [slots['a'], slots['b']] in recipe['sum_pairs']
    return False


def frame_list(recipe):
    """Every frame in its order: statements, then questions; index from 1."""
    out = [(g['id'], g) for g in recipe['statements']] + [(q['id'], q) for q in recipe.get('questions', ())]
    return {fid: i + 1 for i, (fid, _) in enumerate(out)}


# -- inference from what is said (section 14.5) ---------------------------------

def closure(pairs):
    facts = set(pairs)
    while True:
        more = {(a, d) for a, b in facts for c, d in facts if b == c and a != d} - facts
        if not more:
            return facts
        facts |= more


CHOICE = {'greater_of': ('level_before', 1), 'bigger_of': ('bigger', 0), 'first_of': ('precedes', 0)}


def follows(statements, question, cast):
    """What follows from the said statements: a truth, an answer, or None."""
    pred, args = question['predicate'], [tuple(a) if isinstance(a, list) else a for a in question['args']]
    facts = {}
    rules, moved, every = [], set(), []
    for s in statements:
        p, a = s['predicate'], s['args']
        if p in ORDERS:
            facts.setdefault(p, set()).add(tuple(a))
        elif p in ('if_moves_moves', 'if_moves_still'):
            rules.append((p, a[0], a[1]))
        elif p == 'moved':
            moved.add(a[0])
        elif p in ('every_above', 'every_below'):
            every.append((p, a[0], a[1]))
    if pred in ORDERS:
        known = closure(facts.get(pred, ()))
        if '?' in args:
            i = args.index('?')
            found = sorted({f[i] for f in known if f[1 - i] == args[1 - i]})
            if len(found) == 1:
                return found[0]
            return None
        x, y = args
        if (x, y) in known:
            return True
        if (y, x) in known:
            return False
        for p, k, b in every:
            if pred == 'row_before' and b == y and x in cast.members[k]:
                return p == 'every_above'
        return None
    if pred in CHOICE:
        order, keep = CHOICE[pred]
        known = closure(facts.get(order, ()))
        a, b = args
        if (a, b) in known:
            return (a, b)[keep]
        if (b, a) in known:
            return (b, a)[keep]
        return None
    if pred == 'moved':
        for p, a, b in rules:
            if b == args[0] and a in moved:
                return p == 'if_moves_moves'
        return None
    return None


def asked_fact(predicate, args, value):
    """The order fact a statement states or a question asks, as (order, (x, y))."""
    args = [value if a == '?' else a for a in args]
    if predicate in ORDERS:
        return predicate, tuple(args)
    if predicate in CHOICE:
        order, keep = CHOICE[predicate]
        a, b = args
        other = b if value == a else a
        return order, ((other, value) if keep == 1 else (value, other))
    return None


def opposite(spec, wording):
    """The other wording of a converse pair the recipe declares."""
    for pair in spec['converse_pairs']:
        if wording in pair:
            return pair[1 - pair.index(wording)]
    return None


# -- verify --------------------------------------------------------------------

def verify(rows, recipe, cast):
    """Accept a row its recipe produces and whose meanings hold; reject the rest."""
    if not isinstance(cast, Cast):
        cast = Cast(cast)
    frames = {g['id']: g for g in recipe['statements']}
    frames.update({q['id']: q for q in recipe.get('questions', ())})
    accepted, rejected = 0, []
    for row in rows:
        try:
            ok = _verify_row(row, recipe, cast, frames)
        except (KeyError, IndexError, TypeError, ValueError, StopIteration):
            ok = False
        if ok:
            accepted += 1
        else:
            rejected.append(row.get('key'))
    return dict(accepted=accepted, rejected=rejected)


def sentence_of(frames, fid, part):
    frame = frames[fid]
    return frame['sentences'][part] if 'sentences' in frame else frame


def _verify_row(row, recipe, cast, frames):
    if row['category'] != recipe['category'] or row['name'] != recipe['name']:
        return False
    if row['operators'] != recipe['operators'] or row['split'] not in recipe['splits']:
        return False
    entries = row['entries']
    covered = set()
    view = View([frame_of(e['scene']) for e in entries], cast)
    statements = [m for m in row['meaning'] if 'value' not in m]
    for record in row['meaning']:
        frame = sentence_of(frames, record['frame'], record.get('part', 0))
        if record['predicate'] != frame['predicate']:
            return False
        query = frame['answer'].get('slot') if frame.get('answer', {}).get('type') == 'wh' else None
        if record['args'] != json.loads(json.dumps(resolve(frame['args'], record['slots'], query))):
            return False
        if entries[record['entry']]['text'] != render(frame['text'], record['slots'], recipe, cast):
            return False
        covered.add(record['entry'])
        if record['cast'] != instance_cast(frame, record['slots'], recipe, record.get('value')):
            return False
        if 'value' in record:
            spec = frame['answer']
            if row['split'] == 'inference':
                found = follows(statements, record, cast)
                if record['follows'] != found:
                    return False
                if found is None:
                    if row['inference'] != 'open' or row['answer'] != '':
                        return False
                    continue
                value = found
            else:
                value = evaluate(record, view, frame)
            if value is None or value != (tuple(record['value']) if isinstance(record['value'], list)
                                          else record['value']):
                return False
            if record['entry'] != row['question'] or row['answer'] != answer_word(spec, value, cast):
                return False
            if spec['type'] == 'yes_no' and record['truth'] != value:
                return False
        else:
            if row['split'] == 'inference':
                said = said_truth(record)
                if said is False or record['truth'] is not True:
                    return False
            elif evaluate(record, view, frame) != record['truth']:
                return False
    if any(e['text'] and i not in covered for i, e in enumerate(entries)):
        return False
    if row['split'] == 'witness':
        if sum(1 for m in statements if m['truth'] is False) != 1:
            return False
    elif any(m['truth'] is not True for m in statements):
        return False
    return True


def said_truth(record):
    p, a = record['predicate'], record['args']
    if p == 'level_before':
        return said_level(*a)
    return None


# -- the generator ---------------------------------------------------------------

def normal(value):
    """The row as it reads back from its file: tuples become lists."""
    return json.loads(json.dumps(value))


class Generator:
    def __init__(self, recipe, cast, rng):
        self.recipe, self.cast, self.rng = recipe, cast, rng
        self.index = frame_list(recipe)
        self.frames = {g['id']: g for g in recipe['statements']}
        self.frames.update({q['id']: q for q in recipe.get('questions', ())})
        self.seen = set()
        self.short = {}
        self.attempts = 0
        self.said_view = View([None], cast)

    # slots

    def kinds_for(self, slots, constraints, name):
        allowed = list(VALUES)
        for rule, kslot in constraints.get(name, ()):
            members = self.cast.members[slots[kslot]]
            allowed = [d for d in allowed if (d in members) == (rule == 'kind')]
        return allowed

    def assign(self, names, preset, script, avoid=()):
        """Fill slots: kinds first, then digits distinct unless the recipe allows repeats."""
        rng, types = self.rng, self.recipe['slots']
        slots = dict(preset)
        constraints = {}
        for name, rule in sorted(script.get('kinds', {}).items()):
            constraints.setdefault(name, []).extend((r, v) for r, v in sorted(rule.items()))
        for name in names:
            if name not in slots and types.get(name, 'digit') == 'kind':
                slots[name] = rng.choice(self.cast.kind_keys)
        for name in names:
            if name in slots:
                continue
            allowed = self.kinds_for(slots, constraints, name)
            if self.recipe.get('distinct', True):
                taken = {slots[n] for n in names if n in slots and types.get(n, 'digit') == 'digit'}
                allowed = [d for d in allowed if d not in taken]
            if not allowed:
                return None
            pool = [d for d in avoid if d in allowed]
            slots[name] = rng.choice(pool) if pool and rng.random() < 0.75 else rng.choice(allowed)
        for name, rules in constraints.items():
            if any(slots[name] not in self.kinds_for(slots, {name: [r]}, name) for r in rules):
                return None
        return slots

    def names_of(self, frame, script):
        names = []
        for part in frame.get('sentences', [frame]):
            for n in text_slots(part['text'], self.recipe) + [a for a in part['args'] if isinstance(a, str)]:
                if n not in names:
                    names.append(n)
        if frame.get('answer', {}).get('slot') and frame['answer']['slot'] not in names:
            names.append(frame['answer']['slot'])
        for n in sorted(script.get('hidden', ())):
            if n not in names:
                names.append(n)
        return names

    # worlds

    def grow(self, anchor, count):
        shape = {anchor}
        for _ in range(count * 6):
            if len(shape) >= count:
                break
            p = self.rng.choice(sorted(shape))
            q = add(p, self.rng.choice(UNITS))
            if inside(q):
                shape.add(q)
        return frozenset(shape) if len(shape) == count else None

    def propose(self, atom, t, placed):
        """A place for thing t that the atom asks of the things already placed."""
        rng = self.rng
        pred, args = atom
        others = [a for a in args if isinstance(a, int) and a != t]
        if pred == 'between':
            a, b, c = args
            if t == a and b in placed and c in placed:
                pb, pc = min(placed[b]), min(placed[c])
                for axis in (0, 1):
                    if pb[1 - axis] == pc[1 - axis] and abs(pb[axis] - pc[axis]) >= 2:
                        lo, hi = sorted((pb[axis], pc[axis]))
                        k = rng.randint(lo + 1, hi - 1)
                        return (k, pb[1]) if axis == 0 else (pb[0], k)
                return None
            other = next((o for o in others if o in placed), None)
            if other is None:
                return None
            p = min(placed[other])
            unit = rng.choice(UNITS)
            return add(p, unit, rng.randint(1, 3))
        if len(others) != 1 or others[0] not in placed:
            return None
        o = placed[others[0]]
        first = args.index(t) < args.index(others[0])
        px, py = rng.choice(sorted(o))
        if pred in ('row_before', 'on_top'):
            if pred == 'on_top':
                return (px, min(y for x, y in o if x == px) - 1) if first else (px, max(y for x, y in o if x == px) + 1)
            if first:
                return (px, rng.randint(0, min(y for _, y in o) - 1)) if min(y for _, y in o) > 0 else None
            return (px, rng.randint(max(y for _, y in o) + 1, SIDE - 1)) if max(y for _, y in o) < SIDE - 1 else None
        if pred == 'column_before':
            if first:
                return (rng.randint(0, min(x for x, _ in o) - 1), py) if min(x for x, _ in o) > 0 else None
            return (rng.randint(max(x for x, _ in o) + 1, SIDE - 1), py) if max(x for x, _ in o) < SIDE - 1 else None
        if pred in ('adjacent', 'touches'):
            return add((px, py), rng.choice(UNITS))
        if pred == 'next_to':
            return (px + rng.choice((-1, 1)), py)
        if pred == 'near':
            return (px + rng.randint(-NEAR, NEAR), py + rng.randint(-NEAR, NEAR))
        if pred == 'far':
            far = [(x, y) for x in range(SIDE) for y in range(SIDE) if max(abs(x - px), abs(y - py)) >= FAR]
            return rng.choice(far) if far else None
        return None

    def free_place(self, v, taken):
        for _ in range(60):
            p = (self.rng.randrange(SIDE), self.rng.randrange(SIDE))
            if p not in taken and p not in RESERVED[v]:
                return p
        return None

    def place(self, things, sizes, guides, taken):
        rng = self.rng
        placed = {}
        order = list(things)
        rng.shuffle(order)
        order.sort(key=lambda v: -sum(1 for g in guides if v in g[1]))
        busy = set(taken)
        for v in order:
            for _ in range(30):
                anchor = None
                usable = [g for g in guides if v in g[1]
                          and all(a in placed for a in g[1] if isinstance(a, int) and a != v)
                          and any(isinstance(a, int) and a != v for a in g[1])]
                if usable:
                    anchor = self.propose(rng.choice(usable), v, placed)
                if anchor is None:
                    if placed and rng.random() < 0.5:
                        o = min(placed[rng.choice(sorted(placed))])
                        anchor = (o[0], rng.randrange(SIDE)) if rng.random() < 0.5 else (rng.randrange(SIDE), o[1])
                    else:
                        anchor = (rng.randrange(SIDE), rng.randrange(SIDE))
                if not inside(anchor):
                    continue
                shape = self.grow(anchor, sizes[v])
                if shape is None or shape & busy or shape & RESERVED[v]:
                    continue
                placed[v] = shape
                busy |= shape
                break
            else:
                return None
        return placed

    def sizes_for(self, things, hints):
        choice = self.recipe.get('sizes', [1])
        out = {}
        for v in things:
            hint = hints.get(v)
            if hint == 'big':
                out[v] = self.rng.choice([s for s in choice if s >= BIG])
            elif hint == 'small':
                out[v] = 1
            else:
                out[v] = self.rng.choice(choice)
        return out

    def motion(self, specs, timing):
        """A short script of events: each mover's place at every image."""
        rng = self.rng
        lead = rng.randint(0, 2)
        t = lead
        cur, anchors, plan = {}, {}, []
        first_steps = None
        for spec in specs:
            who, to, rel = spec['who'], spec.get('to'), spec.get('rel')
            if who in anchors or (to is not None and to in cur):
                return None
            lo, hi = spec.get('steps', (2, 3))
            steps = first_steps if spec.get('same_steps') and first_steps else rng.randint(lo, hi)
            n = spec.get('speed') or rng.choice((1, 2))
            unit = tuple(spec['dir']) if spec.get('dir') else None
            t0 = lead if timing == 'concurrent' else t
            p0 = cur.get(who)
            if rel in ('toward', 'reach', 'away', 'leave'):
                gap = 1 if rel in ('reach', 'leave') else rng.randint(1, 3)
                ahead = rel in ('toward', 'reach')
                if to in anchors:
                    b = anchors[to]
                    if p0 is None:
                        unit = unit or rng.choice(UNITS)
                        p0 = add(b, unit, -(n * steps + gap)) if ahead else add(b, unit, gap)
                    else:
                        d = (b[0] - p0[0], b[1] - p0[1]) if ahead else (p0[0] - b[0], p0[1] - b[1])
                        if (d[0] == 0) == (d[1] == 0):
                            return None
                        k = abs(d[0] + d[1])
                        unit = (d[0] // k, d[1] // k)
                        if ahead and k - n * steps < 1 or (rel == 'reach' and k - n * steps != 1) \
                                or (rel == 'leave' and k != 1):
                            return None
                else:
                    if p0 is None:
                        p0 = (rng.randrange(SIDE), rng.randrange(SIDE))
                    unit = unit or rng.choice(UNITS)
                    anchors[to] = add(p0, unit, n * steps + gap) if ahead else add(p0, unit, -gap)
            elif spec.get('end') or spec.get('start'):
                pred, x, y = spec.get('end') or spec.get('start')
                other = y if x == who else x
                if other not in anchors:
                    anchors[other] = (rng.randrange(SIDE), rng.randrange(SIDE))
                unit = unit or rng.choice(UNITS)
                p = self.propose((pred, [x, y]), who, {other: frozenset([anchors[other]])})
                if p is None:
                    return None
                start = add(p, unit, -n * steps) if spec.get('end') else p
                if p0 is not None and p0 != start:
                    return None
                p0 = start
            else:
                if p0 is None:
                    p0 = (rng.randrange(SIDE), rng.randrange(SIDE))
                options = [u for u in UNITS if spec.get('dir_not') is None or list(u) != spec['dir_not']]
                unit = unit or rng.choice(options)
            plan.append((who, t0, steps, n, unit, p0))
            cur[who] = add(p0, unit, n * steps)
            first_steps = first_steps or steps
            if timing != 'concurrent':
                t = t0 + steps + rng.randint(1, 2)
        last = max(t0 + steps for _, t0, steps, *_ in plan) + 1
        paths = {}
        for who in dict.fromkeys(p[0] for p in plan):
            mine = [p for p in plan if p[0] == who]
            where = []
            for k in range(last + 1):
                place = mine[0][5]
                for _, t0, steps, n, unit, p0 in mine:
                    if k >= t0:
                        place = add(p0, unit, n * min(k - t0, steps))
                where.append(place)
            if any(not inside(p) or p in RESERVED[who] for p in where):
                return None
            paths[who] = where
        for v, p in anchors.items():
            if not inside(p) or p in RESERVED[v]:
                return None
        return paths, anchors, last

    def world(self, instances, things, extra, falsified):
        """Images for the instances; static unless some frame asks for events.

        ``falsified`` is the index of the one false sentence of the first
        instance (the witness split): the events and layout that serve it are
        left out, and half the time its thing is given an event of no relation.
        """
        rng = self.rng
        specs, guides, hints, timing = [], [], {}, 'sequential'
        movers = set()
        for inst in instances:
            frame, slots, script = inst['frame'], inst['slots'], inst['script']
            false_here = falsified is not None and inst is instances[0]
            events = []
            for spec in script.get('events', ()):
                if false_here and spec.get('part', falsified) == falsified:
                    if rng.random() < 0.5:
                        events.append({'who': spec['who']})
                    continue
                events.append(spec)
            if script.get('timing'):
                timing = script['timing']
            for spec in events:
                spec = dict(spec)
                for key in ('who', 'to'):
                    if key in spec:
                        spec[key] = slots[spec[key]]
                for key in ('end', 'start'):
                    if key in spec:
                        spec[key] = [spec[key][0], slots[spec[key][1]], slots[spec[key][2]]]
                if spec['who'] in movers:
                    continue
                specs.append(spec)
            movers |= {s['who'] for s in specs}
            if not false_here:
                for atom in script.get('layout', ()):
                    guides.append((atom[0], [slots[a] for a in atom[1:]]))
            for name, hint in script.get('size', {}).items():
                hints[slots[name]] = hint
            for part in inst['parts']:
                f = part['frame']
                if f['predicate'] in SAID:
                    continue
                if part['expect'] is True:
                    guides.append((f['predicate'], part['args']))
                elif f.get('answer', {}).get('type') == 'wh':
                    guides.append((f['predicate'], resolve(f['args'], slots)))
        sizes = self.sizes_for(things, hints)
        if specs:
            built = self.motion(specs, timing)
            if built is None:
                return None
            paths, anchors, last = built
            taken = {p for path in paths.values() for p in path} | set(anchors.values())
            if len(set(anchors.values())) != len(anchors) or set(anchors) - set(things):
                return None
            rest = [v for v in things if v not in paths and v not in anchors]
            placed = self.place(rest, {v: sizes[v] for v in rest}, guides, taken)
            if placed is None:
                return None
            frames = []
            for k in range(last + 1):
                f = {v: frozenset([path[k]]) for v, path in paths.items()}
                f.update({v: frozenset([p]) for v, p in anchors.items()})
                f.update(placed)
                if len({p for s in f.values() for p in s}) != sum(len(s) for s in f.values()):
                    return None
                frames.append(f)
            return frames + [frames[-1]] * extra
        placed = self.place(list(things), sizes, guides, set())
        return None if placed is None else [placed]

    # documents

    def instance(self, frame, slots, script, role, expect):
        parts = []
        for i, part in enumerate(frame.get('sentences', [frame])):
            query = part['answer'].get('slot') if part.get('answer', {}).get('type') == 'wh' else None
            parts.append(dict(frame=part, part=i, args=resolve(part['args'], slots, query),
                              text=render(part['text'], slots, self.recipe, self.cast),
                              expect=expect[i] if isinstance(expect, list) else expect))
        return dict(frame=frame, slots=slots, script=script, role=role, parts=parts)

    def check(self, part, view, t, slots):
        frame = part['frame']
        record = dict(entry=t, predicate=frame['predicate'], args=part['args'])
        value = evaluate(record, view, frame)
        if value is None:
            return None
        spec = frame.get('answer')
        if spec is None:
            return value == part['expect']
        if spec['type'] == 'wh':
            if spec.get('free'):
                slots[spec['slot']] = value
            elif value != slots[spec['slot']]:
                return False
        if spec.get('role') and value != slots[spec['role']]:
            return False
        return part['expect'] is None or value == part['expect']

    def entries(self, instances, frames, motion):
        """Put each text at an image where its meaning holds; texts told in order."""
        rng = self.rng
        texts = [(inst, part) for inst in instances for part in inst['parts']]
        stmts = [x for x in texts if x[0]['role'] == 'statement']
        quest = [x for x in texts if x[0]['role'] == 'question']
        arranged = [scene_of(f, rng) if f is not None else None for f in frames]
        if not motion:
            arranged = [scene_of(frames[0], rng) if frames[0] is not None else None for _ in texts]
            view = View([frame_of(s) for s in arranged], self.cast)
            positions = list(range(len(texts)))
            for (inst, part), t in zip(texts, positions):
                if not self.check(part, view, t, inst['slots']):
                    return None
            return arranged, positions
        view = View([frame_of(s) for s in arranged], self.cast)
        valid = [[t for t in range(view.count) if self.check(part, view, t, inst['slots'])]
                 for inst, part in texts]
        for attempt in range(40):
            # the first twenty tries ask the question after the statements; then anywhere
            strict = attempt < 20
            used, positions, last = set(), [], -1
            for (inst, part), options in zip(texts, valid):
                if inst['role'] == 'statement':
                    options = [t for t in options if t > last]
                elif strict:
                    options = [t for t in options if t > last and t not in used]
                else:
                    options = [t for t in options if t not in used]
                if not options:
                    break
                t = rng.choice(options)
                used.add(t)
                positions.append(t)
                if inst['role'] == 'statement':
                    last = t
            else:
                break
        else:
            return None
        if len(positions) != len(texts):
            return None
        keep = max(max(positions) + 1, view.count - 3)
        arranged = arranged[:keep]
        view = View([frame_of(s) for s in arranged], self.cast)
        for (inst, part), t in zip(texts, positions):
            if not self.check(part, view, t, inst['slots']):
                return None
        return arranged, positions

    def row(self, split, instances, arranged, positions, variant, inference=None, contrast=None):
        entries = [dict(text='', scene=s) for s in arranged]
        meaning, question, answer = [], None, None
        texts = [(inst, part) for inst in instances for part in inst['parts']]
        view = View([frame_of(s) for s in arranged], self.cast)
        for (inst, part), t in zip(texts, positions):
            entries[t]['text'] = part['text']
            frame = part['frame']
            record = dict(entry=t, frame=inst['frame']['id'], slots=self.visible(inst, frame),
                          predicate=frame['predicate'], args=part['args'])
            if 'sentences' in inst['frame']:
                record['part'] = part['part']
            if frame.get('answer'):
                value = part.get('value')
                if value is None:
                    value = evaluate(record, view, frame)
                record['value'] = value
                record['truth'] = value if frame['answer']['type'] == 'yes_no' else True
                if inference:
                    record['follows'] = part.get('follows')
                question = t
                answer = part.get('answer_word', None)
                if answer is None:
                    answer = answer_word(frame['answer'], value, self.cast)
            else:
                record['truth'] = part['expect'] if part['expect'] is not None else True
            record['cast'] = instance_cast(frame, record['slots'], self.recipe, record.get('value'))
            meaning.append(record)
        identity = json.dumps(entries, sort_keys=True)
        if identity in self.seen:
            return None
        self.seen.add(identity)
        return normal(dict(category=self.recipe['category'], name=self.recipe['name'], variant=variant,
                    split=split, operators=self.recipe['operators'], entries=entries,
                    meaning=meaning, question=question, answer=answer,
                    inference=inference, set=contrast))

    def visible(self, inst, frame):
        names = text_slots(frame['text'], self.recipe) + [a for a in frame['args'] if isinstance(a, str)]
        if frame.get('answer', {}).get('slot'):
            names.append(frame['answer']['slot'])
        return {n: inst['slots'][n] for n in sorted(set(names))}

    def compose(self, split, group, question, target):
        """One document: a statement group and a question, true of drawn images."""
        rng, recipe = self.rng, self.recipe
        witness = split == 'witness'
        general = split == 'generalization'
        gscript = group.get('script', {}) if group else {}
        for _ in range(ATTEMPTS):
            self.attempts += 1
            instances, names = [], []
            preset = {}
            false_part = None
            if general and recipe['held_out'] == 'pairs':
                carrier = rng.choice([f for f in (group, question) if f and
                                      len([s for s in self.text_digits(f)]) >= 2] or [None])
                if carrier is None:
                    return None
                p, q = rng.choice(HELD_OUT_PAIRS)
                if rng.random() < 0.5:
                    p, q = q, p
                first, second = rng.sample(self.text_digits(carrier), 2)
                preset[carrier['id']] = {first: p, second: q}
            elif general and recipe['held_out'] == 'frame_digit':
                carrier = rng.choice([f for f in (group, question) if f and self.text_digits(f)] or [None])
                if carrier is None:
                    return None
                slot = rng.choice(self.text_digits(carrier))
                d = rng.choice([d for d in VALUES if (self.index[carrier['id']] + d) % 4 == 0])
                preset[carrier['id']] = {slot: d}
            elif general and recipe['held_out'] == 'sums':
                carrier = rng.choice([f for f in (group, question) if f and self.sums(f)] or [None])
                if carrier is None:
                    return None
                a, b = rng.choice(recipe['sum_pairs'])
                preset[carrier['id']] = {'a': a, 'b': b, 'c': a + b}
            gslots = None
            if group:
                fixed = dict(group.get('fixed', {}))
                fixed.update(preset.get(group['id'], {}))
                gslots = self.assign(self.names_of(group, gscript), fixed, gscript)
                if gslots is None:
                    continue
                n_parts = len(group.get('sentences', [group]))
                expect = [True] * n_parts
                if witness:
                    candidates = [i for i, p in enumerate(group.get('sentences', [group]))
                                  if p['predicate'] not in SAID]
                    if not candidates:
                        return None
                    false_part = rng.choice(candidates)
                    expect[false_part] = False
                inst = self.instance(group, gslots, gscript, 'statement', expect)
                instances.append(inst)
            if question:
                qspec = question['answer']
                qscript = question.get('script', {})
                if isinstance(qscript, dict) and qspec['type'] in ('yes_no',) and 'yes' in qscript:
                    qscript = qscript['yes' if target else 'no']
                elif isinstance(qscript, dict) and 'by_target' in qscript:
                    qscript = qscript['by_target'][str(target)]
                fixed = dict(preset.get(question['id'], {}))
                for name in question.get('bind', ()):
                    fixed[name] = gslots[name]
                # the question speaks of the statement's things where it can, hidden ones included
                pool = sorted({v for k, v in (gslots or {}).items() if recipe['slots'].get(k, 'digit') == 'digit'})
                rng.shuffle(pool)
                qslots = self.assign(self.names_of(question, qscript), fixed, qscript, avoid=pool)
                if qslots is None:
                    continue
                if qspec.get('roles'):
                    first, second = qspec['roles']
                    if rng.random() < 0.5:
                        first, second = second, first
                    qslots['@answer'], qslots['@other'] = qslots[first], qslots[second]
                    qspec = dict(qspec, role='@answer')
                    question = dict(question, answer=qspec)
                inst = self.instance(question, qslots, qscript, 'question', target)
                instances.append(inst)
            clean = True
            if general != self.holds_out(instances):
                continue
            # said meanings hold of the slots alone
            for inst in instances:
                for part in inst['parts']:
                    p = part['frame']['predicate']
                    if p in PRUNE and inst['role'] == 'statement':
                        if PREDICATES[p](self.said_view, 0, *part['args']) is not part['expect']:
                            clean = False
            if not clean:
                continue
            if any(p['frame'].get('image') is False for i in instances for p in i['parts']) \
                    or recipe.get('world') == 'none':
                arranged = [None] * sum(len(i['parts']) for i in instances)
                view = View(arranged, self.cast)
                ok = all(self.check(p, view, t, i['slots']) for t, (i, p) in
                         enumerate((i, p) for i in instances for p in i['parts']))
                if not ok or general != self.holds_out(instances):
                    continue
                return instances, arranged, list(range(len(arranged)))
            things = set()
            absent = set()
            for inst in instances:
                script = inst['script'] if inst is not None else {}
                if witness and inst['role'] == 'statement':
                    script = {k: v for k, v in script.items() if k not in ('absent', 'one_of')}
                for name in script.get('absent', ()):
                    absent.add(inst['slots'][name])
                for pair in script.get('one_of', ()):
                    drop = rng.choice(pair)
                    absent.add(inst['slots'][drop])
                for k, v in inst['slots'].items():
                    if recipe['slots'].get(k, 'digit') == 'digit' and not k.startswith('@'):
                        things.add(v)
            if witness:
                for part in instances[0]['parts']:
                    if part['expect'] is False:
                        for a in part['args']:
                            if isinstance(a, int) and rng.random() < 0.25:
                                absent.add(a)
            things -= absent
            only = any(i['script'].get('only') for i in instances)
            if not only:
                spare = [d for d in VALUES if d not in things and d not in absent]
                rng.shuffle(spare)
                things |= set(spare[:rng.choice((0, 0, 1, 2))])
            things = sorted(things)
            frames = self.world(instances, things, 3, false_part if witness else None)
            if frames is None:
                continue
            motion = len(frames) > 1
            placed = self.entries(instances, frames, motion)
            if placed is None:
                continue
            arranged, positions = placed
            if general != self.holds_out(instances):
                continue
            return instances, arranged, positions
        return None

    def holds_out(self, instances):
        """Whether any frame instance names a cast held out of this category."""
        for inst in instances:
            slots = {k: v for k, v in inst['slots'].items() if not k.startswith('@')}
            for part in inst['parts']:
                f = part['frame']
                spec = f.get('answer')
                value = slots[spec['slot']] if spec and spec['type'] == 'wh' else None
                if held_out(self.recipe, self.index[inst['frame']['id']], f, slots,
                            instance_cast(f, slots, self.recipe, value)):
                    return True
        return False

    def text_digits(self, frame):
        names = []
        for part in frame.get('sentences', [frame]):
            for n in named_slots(part, self.recipe):
                if self.recipe['slots'].get(n) == 'digit' and n not in names:
                    names.append(n)
        return names

    def sums(self, frame):
        return any(p['predicate'] == 'sum' for p in frame.get('sentences', [frame]))

    def ordinary(self, split, want):
        recipe, rng = self.recipe, self.rng
        groups = [g for g in recipe['statements'] if not g.get('inference_only')]
        questions = [q for q in recipe.get('questions', ()) if not q.get('inference_only')]
        if split == 'witness':
            groups = [g for g in groups if not g.get('no_witness') and any(
                p['predicate'] not in SAID and p.get('text') for p in g.get('sentences', [g]))]
        if recipe.get('compose') == 'statement':
            questions = []
        if recipe.get('compose') == 'question':
            groups = []
        made, counts, failures = [], {}, 0
        gcycle, qcycle = [], []
        # A recipe may divide its groups into halves (category 20: the neutral
        # tellings and the worded ones): each half gets half the documents of a
        # split, and the answers are balanced within each half.
        halves = [[g for g in groups if g['id'] in ids] for ids in recipe.get('halves', ())]
        halves = [h for h in halves if h]
        hcycles = [[] for _ in halves]
        made_in = [0] * len(halves)
        half = None
        while len(made) < want and failures < 3 * want + 30:
            if halves:
                half = min(range(len(halves)), key=lambda h: (made_in[h], h))
                if not hcycles[half]:
                    hcycles[half] = list(halves[half])
                    rng.shuffle(hcycles[half])
            elif groups and not gcycle:
                gcycle = list(groups)
                rng.shuffle(gcycle)
            if questions and not qcycle:
                qcycle = list(questions)
                rng.shuffle(qcycle)
            if halves:
                group = hcycles[half].pop()
            else:
                group = gcycle.pop() if groups else None
            question = qcycle.pop() if questions else None
            target = None
            if question:
                spec = question['answer']
                if spec['type'] == 'yes_no':
                    values = [True, False]
                elif 'words' in spec:
                    values = list(range(len(spec['words'])))
                else:
                    values = None
                if values:
                    key = (lambda v: str(v)) if half is None else (lambda v: f'{half}:{v}')
                    target = min(values, key=lambda v: (counts.get(key(v), 0), rng.random()))
            built = self.compose(split, group, question, target)
            if built is None:
                failures += 1
                continue
            instances, arranged, positions = built
            variant = '+'.join(i['frame']['id'] for i in instances)
            row = self.row(split, instances, arranged, positions, variant)
            if row is None:
                failures += 1
                continue
            if target is not None:
                tkey = str(target) if half is None else f'{half}:{target}'
                counts[tkey] = counts.get(tkey, 0) + 1
            if half is not None:
                made_in[half] += 1
            made.append(row)
        return made

    def counting(self, split, want):
        rng = self.rng
        frame = self.recipe['statements'][0]
        made, failures = [], 0
        while len(made) < want and failures < 3 * want + 30:
            n = rng.randint(1, 9)
            values = sorted(rng.sample(VALUES, n))
            if (sum(values) % 4 == 0) != (split == 'generalization'):
                failures += 1
                continue
            placed = {}
            for v in values:
                p = self.free_place(v, set(placed.values()))
                if p is None:
                    break
                placed[v] = p
            if len(placed) != n:
                failures += 1
                continue
            arranged = [scene_of({v: frozenset([p]) for v, p in placed.items()}, rng)]
            inst = self.instance(frame, {}, {}, 'question', None)
            row = self.row(split, [inst], arranged, [0], frame['id'])
            if row is None:
                failures += 1
                continue
            made.append(row)
        return made

    def reserved(self, split, want):
        """Category 2's generalization: each value at its reserved step and place."""
        rng = self.rng
        frame = self.frames[self.recipe['reserved_question']]
        made, failures = [], 0
        while len(made) < want and failures < 3 * want + 30:
            d = rng.choice(VALUES)
            p = rng.choice(sorted(RESERVED[d]))
            arranged = [scene_of({d: frozenset([p])}, rng, reserved=True)]
            inst = self.instance(frame, {frame['answer']['slot']: d}, {}, 'question', None)
            row = self.row(split, [inst], arranged, [0], frame['id'])
            if row is None:
                failures += 1
                continue
            made.append(row)
        return made

    def contrast(self, want):
        """Sets of twelve: a above b and b above a, three questions each, and both shifted."""
        rng, spec = self.rng, self.recipe['contrast']
        trained = [(a, b) for a in VALUES for b in VALUES if a != b
                   and (a, b) not in HELD_OUT_PAIRS and (b, a) not in HELD_OUT_PAIRS]
        held = [p for pair in HELD_OUT_PAIRS for p in (pair, pair[::-1])]
        pairs = rng.sample(trained, want // 2) + [rng.choice(held) for _ in range(want - want // 2)]
        made = []
        for number, (a, b) in enumerate(pairs):
            for _ in range(500):
                x = rng.randrange(SIDE)
                y1, y2 = sorted(rng.sample(range(SIDE), 2))
                dx, dy = rng.choice([(i, j) for i in range(-3, 4) for j in range(-3, 4) if (i, j) != (0, 0)])
                spots = [(x, y1), (x, y2), (x + dx, y1 + dy), (x + dx, y2 + dy)]
                if all(inside(p) and p not in RESERVED[a] and p not in RESERVED[b] for p in spots):
                    break
            else:
                continue
            key = f"set-{self.recipe['category']:02d}-{number:02d}"
            rows = []
            for offset in ((0, 0), (dx, dy)):
                top, bottom = add((x, y1), offset), add((x, y2), offset)
                for upper, lower in ((a, b), (b, a)):
                    arrangement = {upper: frozenset([top]), lower: frozenset([bottom])}
                    for qid, slots in spec['questions']:
                        frame = self.frames[qid]
                        values = {'a': a, 'b': b}
                        qslots = {k: values[v] for k, v in slots.items()}
                        arranged = [scene_of(arrangement, rng)]
                        inst = self.instance(frame, qslots, {}, 'question', None)
                        row = self.row('contrast', [inst], arranged, [0], qid, contrast=key)
                        if row is None:
                            break
                        rows.append(row)
            if len(rows) == 12:
                made.extend(rows)
        return made

    def inference(self, want):
        spec = self.recipe['inference']
        rule = spec['rule']
        if rule == 'order':
            return self.infer_order(spec, want)
        if rule == 'if':
            return self.infer_rule(spec, want)
        return self.infer_kind(spec, want)

    def said_doc(self, premises, question, kind, follows_value):
        """Premises and a question with no image."""
        instances = []
        for frame, slots in premises:
            instances.append(self.instance(frame, slots, {}, 'statement', True))
        frame, slots = question
        inst = self.instance(frame, slots, {}, 'question', None)
        part = inst['parts'][0]
        part['follows'] = follows_value
        part['value'] = follows_value
        if follows_value is None:
            part['answer_word'] = ''
        instances.append(inst)
        count = sum(len(i['parts']) for i in instances)
        return self.row('inference', instances, [None] * count, list(range(count)),
                        '+'.join(i['frame']['id'] for i in instances), inference=kind)

    def infer_order(self, spec, want):
        rng = self.rng
        made, counts, failures = [], {}, 0
        kinds = spec['types']
        orders = sorted({self.frames[i]['predicate'] for i in spec['premises']})
        while len(made) < want and failures < 30 * want:
            # converse and chain alternate, so that each is half the split
            kind = kinds[len(made) % len(kinds)] if spec.get('alternate') else rng.choice(kinds)
            order = rng.choice(orders)
            premises = [self.frames[i] for i in spec['premises'] if self.frames[i]['predicate'] == order]
            questions = [self.frames[i] for i in spec['questions']
                         if order in (self.frames[i]['predicate'], CHOICE.get(self.frames[i]['predicate'], ('',))[0])]
            chain = kind in ('chain', 'time')
            names = rng.sample(VALUES, 3 if chain else 2)
            if spec.get('numeric'):
                names.sort()
            facts = [(names[0], names[1])] + ([(names[1], names[2])] if chain else [])
            if kind == 'converse':
                # the other way round: one premise in one wording of a pair, asked in the other
                asked = {q.get('wording') for q in questions}
                worded = [f for f in premises if f.get('wording') and opposite(spec, f['wording']) in asked]
                if not worded:
                    failures += 1
                    continue
                premises = [rng.choice(worded)]
                other = opposite(spec, premises[0]['wording'])
                questions = [q for q in questions if q.get('wording') == other]
                if not questions:
                    failures += 1
                    continue
            said = []
            for fact in facts:
                frame = rng.choice(premises)
                slot_names = [a for a in frame['args'] if isinstance(a, str)]
                said.append((frame, dict(zip(slot_names, fact))))
            frame = rng.choice(questions)
            ends = (names[0], names[2]) if chain else (names[0], names[1])
            pair = list(ends)
            rng.shuffle(pair)
            qnames = [a for a in frame['args'] if isinstance(a, str)]
            aspec = frame['answer']
            if aspec['type'] == 'wh':
                others = [n for n in qnames if n != aspec['slot']]
                qslots = {others[0]: pair[0]}
            else:
                qslots = dict(zip(qnames, pair))
            query = aspec['slot'] if aspec['type'] == 'wh' else None
            qrecord = dict(predicate=frame['predicate'], args=resolve(frame['args'], qslots, query))
            srecords = [dict(predicate=f['predicate'], args=resolve(f['args'], s)) for f, s in said]
            value = follows(srecords, qrecord, self.cast)
            if value is None:
                failures += 1
                continue
            if aspec['type'] == 'wh':
                qslots[aspec['slot']] = value
            # while yes and no are unequal, the next row is the one behind, so the split ends even
            ahead = counts.get('True', 0) - counts.get('False', 0)
            if ahead and (aspec['type'] != 'yes_no' or (value is True) == (ahead > 0)):
                failures += 1
                continue
            fact = asked_fact(frame['predicate'], qrecord['args'], value)
            if any(asked_fact(f['predicate'], r['args'], None) == fact and f.get('wording') == frame.get('wording')
                   for (f, _), r in zip(said, srecords)):
                failures += 1
                continue
            if chain and set(ends) != set(pair):
                failures += 1
                continue
            row = self.said_doc(said, (frame, qslots), 'time' if kind == 'time' else kind, value)
            if row is None:
                failures += 1
                continue
            if aspec['type'] == 'yes_no':
                counts[str(value)] = counts.get(str(value), 0) + 1
            made.append(row)
        return made

    def infer_rule(self, spec, want):
        rng = self.rng
        made, failures = [], 0
        plan = [True] * spec['counts']['yes'] + [False] * spec['counts']['no'] + [None] * spec['counts']['open']
        for outcome in plan:
            for _ in range(200):
                a, b = rng.sample(VALUES, 2)
                if outcome is None:
                    rule = self.frames[rng.choice(spec['rules'])]
                else:
                    rule = next(self.frames[r] for r in spec['rules']
                                if (self.frames[r]['predicate'] == 'if_moves_moves') == outcome)
                said = [(rule, {'a': a, 'b': b})]
                if outcome is not None:
                    said.append((self.frames[spec['premise']], {'a': a}))
                row = self.said_doc(said, (self.frames[spec['question']], {'b': b}),
                                    'rule' if outcome is not None else 'open', outcome)
                if row is not None:
                    made.append(row)
                    break
            else:
                failures += 1
        return made

    def infer_kind(self, spec, want):
        rng = self.rng
        made, failures = [], 0
        while len(made) < want and failures < 30 * want:
            outcome = len(made) % 2 == 0
            k = rng.choice(self.cast.kind_keys)
            members = sorted(self.cast.members[k])
            outside = [d for d in VALUES if d not in self.cast.members[k]]
            a, b = rng.choice(members), rng.choice(outside)
            premise = next(self.frames[p] for p in spec['premises']
                           if (self.frames[p]['predicate'] == 'every_above') == outcome)
            row = self.said_doc([(premise, {'k': k, 'b': b})],
                                (self.frames[spec['question']], {'a': a, 'b': b}), 'kind', outcome)
            if row is None:
                failures += 1
                continue
            made.append(row)
        return made

    def split(self, split, want):
        recipe = self.recipe
        if split == 'contrast':
            return self.contrast(want)
        if split == 'inference':
            return self.inference(want)
        if split == 'generalization' and recipe['held_out'] == 'reserved':
            return self.reserved(split, want)
        if recipe.get('compose') == 'count':
            return self.counting(split, want)
        return self.ordinary(split, want)


# -- freeze and load -----------------------------------------------------------

def category_path(recipe):
    return f"category-{recipe['category']:02d}-{recipe['name']}.jsonl"


def words_of(text):
    return re.findall(r'[a-z]+', text)


def freeze(recipes_dir, out_dir, *, seed):
    """Write data/felt/category-NN-<name>.jsonl and manifest.json; tests never call this."""
    recipes_dir, out_dir = Path(recipes_dir), Path(out_dir)
    cast_text = (recipes_dir / 'cast.json').read_text()
    cast = Cast(json.loads(cast_text))
    files, short = [], {}
    longest_words = longest_bytes = 0
    used = set()
    out_dir.mkdir(parents=True, exist_ok=True)
    for recipe_path in sorted(recipes_dir.glob('category-*.json')):
        started = time.time()
        recipe_text = recipe_path.read_text()
        recipe = json.loads(recipe_text)
        gen = Generator(recipe, cast, random.Random(int(seed) * 100 + recipe['category']))
        rows = []
        for split in SPLITS:
            want = recipe['splits'].get(split, 0)
            if not want:
                continue
            got = gen.split(split, want)
            for i, row in enumerate(got):
                row['key'] = f"felt-{recipe['category']:02d}-{split}-{i:04d}"
            rows.extend(got)
            have = len(got) if split != 'contrast' else len(got) // 12
            if have < want:
                short.setdefault(recipe_path.name, {})[split] = [have, want]
        check = verify(rows, recipe, cast)
        if check['rejected']:
            raise ValueError(f'{recipe_path.name}: rows the recipe does not produce: {check["rejected"][:5]}')
        for row in rows:
            for e in row['entries']:
                longest_words = max(longest_words, len(e['text'].split()))
                longest_bytes = max(longest_bytes, len(e['text'].encode()))
                used.update(words_of(e['text']))
            if row['answer']:
                used.update(words_of(row['answer']))
        path = out_dir / category_path(recipe)
        text = ''.join(json.dumps(row, sort_keys=True, separators=(',', ':')) + '\n' for row in rows)
        path.write_text(text)
        counts = {}
        for row in rows:
            counts[row['split']] = counts.get(row['split'], 0) + 1
        files.append(dict(category=recipe['category'], name=recipe['name'], path=path.name,
                          sha256=hashlib.sha256(text.encode()).hexdigest(), rows=counts,
                          verify=dict(accepted=check['accepted'], rejected=0),
                          recipe=f'recipes/{recipe_path.name}',
                          recipe_sha256=hashlib.sha256(recipe_text.encode()).hexdigest(),
                          attempts=gen.attempts))
        print(path.name, counts, f'{time.time() - started:.1f}s', flush=True)
    vocabulary = sorted(cast.data['vocabulary'])
    missing = sorted(used - set(vocabulary))
    if missing:
        raise ValueError(f'words outside the declared vocabulary: {missing}')
    manifest = dict(
        schema=1, generator='bin/felt_world_corpus.py', seed=int(seed),
        command=f'python bin/felt_world_corpus.py freeze --seed {int(seed)} '
                f'--recipes data/felt/recipes --out data/felt',
        cast='recipes/cast.json', cast_sha256=hashlib.sha256(cast_text.encode()).hexdigest(),
        files=files, short=short, vocabulary=vocabulary,
        longest_text=dict(words=longest_words, bytes=longest_bytes),
        reserved={str(d): dict(step=reserved_step(d), places=sorted([list(p) for p in RESERVED[d]]))
                  for d in VALUES},
        held_out_pairs=[list(p) for p in HELD_OUT_PAIRS],
        never_direct='rows are sentences of words beside images of pixel steps; '
                     'no digit, numeric code or readout reaches a text')
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')
    return manifest


@dataclass(frozen=True)
class FeltDocument:
    key: str
    category: int
    split: str
    sentences: tuple
    scenes: tuple
    question: int | None
    answer: str | None
    variant: str
    inference: str | None
    contrast_set: str | None


def load(category, split, *, felt_dir=None):
    """The frozen documents of one category and split; no model is involved."""
    felt_dir = FELT if felt_dir is None else Path(felt_dir)
    manifest = json.loads((felt_dir / 'manifest.json').read_text())
    entry = next((f for f in manifest['files'] if f['category'] == int(category)), None)
    if entry is None:
        raise ValueError(f'unknown felt category: {category}')
    payload = (felt_dir / entry['path']).read_bytes()
    if hashlib.sha256(payload).hexdigest() != entry['sha256']:
        raise ValueError(f'{entry["path"]}: file hash mismatch')
    if split not in entry['rows']:
        raise ValueError(f'category {category} has no {split} split')
    out = []
    for line in payload.decode().splitlines():
        row = json.loads(line)
        if row['split'] != split:
            continue
        out.append(FeltDocument(
            key=row['key'], category=row['category'], split=row['split'],
            sentences=tuple(e['text'] for e in row['entries']),
            scenes=tuple(None if e['scene'] is None else tuple(tuple(p) for p in e['scene'])
                         for e in row['entries']),
            question=row['question'], answer=row['answer'], variant=row['variant'],
            inference=row['inference'], contrast_set=row['set']))
    return tuple(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Freeze the felt-world corpus.')
    parser.add_argument('command', choices=['freeze'])
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--recipes', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    freeze(args.recipes, args.out, seed=args.seed)
