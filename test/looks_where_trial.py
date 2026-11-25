"""A trial in tests only: a row of the first Pi reads the where and the when
of its reading (sigma-pi §19.115, step 20 of round 6).

Built on ``looks_round2`` (the lexicon, the bank, the request, the context
of the taken words). What changes is who reads the where:

- *The way up.* A row of the first Pi reads, beside its members' whats, one
  where and one when for its reading: the band of the event at the
  reading's start (``Trial.theta`` at the token's first place) and the
  comparator's code of the reading's time (``RowTime.how_far_back`` of the
  present against itself: every event of one sentence is read at one
  present). Each wire is an entry of the row on two lanes, the wire's value
  ``c`` and its complement ``1 - c``. The row is ``b = exp(W log m + E+ .
  log c + E- . log(1 - c))``. The entries ``E+``, ``E-`` are learned, start
  at nothing and stay at zero or more. No concept has a weight on a wire.
- *The salience map.* The Sigmas are passed by their weights, the Pi's what
  members by the derivative of its product (as in ``looks_round2``). The
  where and when wires are passed in one of three ways, for measure (i):
  ``'entries'``, the row's entries as a Sigma's weights, the share on a
  wire being ``d_b * b * E * value`` (each lane its own wire);
  ``'lanes'``, the product's derivative with each lane its own wire
  (``value * db/dvalue = d_b * b * E``); ``'combined'``, the product's
  derivative taken on the wire's value ``c``, both lanes through it
  (round 2's row on the where wires).
- *The look.* The what match is the share capped at the event's own
  activity; the where match is ``exp(s - max s)`` over the readings, ``s``
  the where share of a reading's band; it multiplies a reading's word
  values once.
- *The reading lesson* (rule 38, rule 70). Minimal sentences first: one
  word, two words, then the lesson's sixteen. At each look the next unread
  word is booked ``-log softmax(10 * returned)`` over the unread words; the
  look takes the lesson's word; the done mark lowers it. Adam over the
  request's map and the entries.
"""
from dataclasses import dataclass
import math
import os

import torch

from looks_prototype import LESSON
from looks_round2 import (CONTEXT, DTYPE, EPS, G, NOT_SPACE, Bank, Request, TokenMax, Trial, context,
                          interpret, lexicon, minimal_sentences)

N_WHERE = 4

HELD_OUT = ('is This Felix?', 'Felix is here.', 'Ciao is that.', 'is that Ciao?', 'here is Felix.',
            'Felix is there.')


def when_code(n=4):
    """The comparator's code of a reading's time: every event of a sentence
    is read at the one present, so it is the code of no distance back."""
    from RowTime import encoder, how_far_back
    present = encoder().encode(torch.tensor([5_000_000]))
    return how_far_back(present, present)[0].to(DTYPE)


class PiWhere(Trial):
    """The first Pi reads its input whole: what, where and when."""

    def __init__(self, lex, *, forward=True, when=True):
        super().__init__(lex)
        # ``when=False`` leaves the when wires out (an ablation).
        self.when = when_code() if when else torch.zeros(0, dtype=DTYPE)
        self.D = N_WHERE + self.when.numel()
        # Each row's entries on the wires, on two lanes; at nothing.
        self.Ep = torch.zeros(lex.R, self.D, dtype=DTYPE)
        self.Em = torch.zeros(lex.R, self.D, dtype=DTYPE)
        # ``forward=False`` leaves the wires out of the row's value on the way
        # up (an ablation: the entries then act only through the salience map).
        self.forward = forward

    def wires(self, bank):
        band = self.theta[bank.place_of[torch.arange(len(bank.spans)), bank.head_rows]]
        return torch.cat([band, self.when.expand(len(bank.spans), -1)], -1)   # [K, D]

    def up_whole(self, bank, activity, *, Ep=None, Em=None, wires=None, squash=True):
        lex, K = self.lex, len(bank.spans)
        Ep = self.Ep if Ep is None else Ep
        Em = self.Em if Em is None else Em
        wires = self.wires(bank) if wires is None else wires
        act = torch.tanh if squash else (lambda v: v)
        m = TokenMax.apply(activity, bank.token * lex.R + bank.row, K * lex.R).reshape(K, lex.R)
        clamped = m.clamp_min(EPS)
        log_what = clamped.log() @ lex.W.T
        c = wires.clamp(EPS, 1 - EPS)
        log_wire = c.log() @ Ep.T + (1 - c).log() @ Em.T          # [K, R]
        b = (log_what + (log_wire if self.forward else 0.)).exp()
        xw = 1 - (-b).exp()
        cw = act(G * xw)
        pre_words = G * cw[:, lex.word_rows].sum(-1)
        if lex.space is not None:
            pre_words = pre_words - NOT_SPACE * cw[:, lex.space]
        return dict(m=m, clamped=clamped, b=b, cw=cw, words=act(pre_words), wires=wires)

    def shares(self, bank, activity, request, *, way='entries', Ep=None, Em=None):
        """The what share of each event and the where and when shares of each
        reading, by the transposes with each Sigma passed by its weights."""
        lex, K, R = self.lex, len(bank.spans), self.lex.R
        Ep = self.Ep if Ep is None else Ep
        Em = self.Em if Em is None else Em
        out = self.up_whole(bank, activity, Ep=Ep, Em=Em, squash=False)
        # The request on its for lane: salience is never below zero, and what
        # tells against a present word is nothing, so an asking below zero
        # makes nothing salient here (rule 36).
        words = request.words.clamp_min(0.) if torch.is_tensor(request.words) else max(request.words, 0.)
        d_cw = activity.new_zeros(K, R)
        if request.concepts is not None:
            d_cw = d_cw + request.concepts.clamp_min(0.)
        d_cw[:, lex.word_rows] = d_cw[:, lex.word_rows] + G * words
        if lex.space is not None:
            d_cw[:, lex.space] = d_cw[:, lex.space] - NOT_SPACE * words
        b = out['b']
        d_b = G * d_cw * (-b).exp()
        tok, row = bank.token, bank.row
        d_m = ((d_b * b) @ lex.W) / out['clamped'] * (out['m'] >= EPS)
        at_max = (activity == out['m'][tok, row]).to(activity.dtype)
        what = activity * d_m[tok, row] * at_max
        c = out['wires'].clamp(EPS, 1 - EPS)
        gate = d_b * b                                               # [K, R], by presence
        if way == 'entries':
            wire = c[:, None, :] * Ep + (1 - c)[:, None, :] * Em      # [K, R, D]
        elif way == 'lanes':
            wire = Ep + Em + 0 * c[:, None, :]
        else:   # 'combined': c * d/dc of E+ log c + E- log(1 - c)
            wire = c[:, None, :] * (Ep / c[:, None, :] - Em / (1 - c)[:, None, :])
        per_wire = (gate[..., None] * wire).sum(1)                   # [K, D]
        return what, per_wire[:, :N_WHERE].sum(-1), per_wire[:, N_WHERE:].sum(-1), out


def look(trial, bank, request, *, activity=None, way='entries', Ep=None, Em=None, match=True):
    a0 = bank.live() if activity is None else activity
    what, s, when_share, _ = trial.shares(bank, a0, request, way=way, Ep=Ep, Em=Em)
    asks = (torch.is_tensor(request.words) and request.words.requires_grad) or bool(request.words)
    own = torch.where(a0 > 0, torch.minimum(a0, what.clamp_min(0.)) / a0.clamp_min(EPS), 0.) if asks \
        else torch.ones_like(a0)
    live = torch.zeros(len(bank.spans), dtype=DTYPE).index_add(0, bank.token, a0) > 0
    s = torch.where(live, s, s.detach().min() if live.any() else s)
    where = (s - s.max()).exp() if match else torch.ones_like(s)
    out = trial.up_whole(bank, a0 * own, Ep=Ep, Em=Em)
    word = out['cw'][:, trial.lex.word_rows] * where[:, None]
    return dict(word=word, where=where, where_share=s, when_share=when_share, own=own)


class Reader:
    """The request: a learned map from the context to "words" and every row's
    concept, starting at "words" alone (with a little noise of its own)."""

    def __init__(self, trial, generator, noise=1e-2):
        R = trial.lex.R
        self.A = noise * torch.randn(R + 1, CONTEXT, generator=generator, dtype=DTYPE)
        self.A[0, 0] = 1.
        self.A.requires_grad_()

    def request(self, ctx, A=None):
        m = (self.A if A is None else A) @ ctx
        return Request(words=m[0], concepts=m[1:])


def read(trial, reader, bank, *, train=False, sharpness=10., Ep=None, Em=None, match=True,
         way='entries', still=False):
    """Read a sentence look by look; the loss and whether it was read in order.
    ``still`` holds the request at its context of no word read."""
    order = bank.word_tokens()
    activity = bank.live()
    loss, ok, taken = torch.zeros((), dtype=DTYPE), True, []
    for t, target in enumerate(order):
        ctx = context(trial, bank, [] if still else taken)
        out = look(trial, bank, reader.request(ctx), activity=activity, Ep=Ep, Em=Em, match=match,
                   way=way)
        score = out['word'].max(-1).values[order[t:]]
        loss = loss - torch.log_softmax(sharpness * score, 0)[0]
        got = interpret(trial, bank, out)
        ok = ok and got.identified and got.token == target
        taken.append(target)
        activity = activity * (bank.token != target).to(DTYPE)
    return loss, ok


def count(trial, reader, banks, **kw):
    with torch.no_grad():
        return sum(read(trial, reader, bank, **kw)[1] for bank in banks)


def fresh_seed():
    """A start drawn from the system's entropy, printed by the caller."""
    return int.from_bytes(os.urandom(6), 'little')


def teach(*, seed=None, stages=(10, 10, 60), rate=.05, way='entries', match=True, forward=True,
          when=True, learn_entries=True, learn_request=True):
    """One start of the reading lesson. Returns a record of the measures."""
    seed = fresh_seed() if seed is None else seed
    generator = torch.Generator().manual_seed(seed)
    trial = PiWhere(lexicon(), forward=forward, when=when)
    reader = Reader(trial, generator)
    Ep = trial.Ep.clone().requires_grad_(learn_entries)
    Em = trial.Em.clone().requires_grad_(learn_entries)
    groups = ([reader.A] if learn_request else []) + ([Ep, Em] if learn_entries else [])
    optimizer = torch.optim.Adam(groups, lr=rate)
    one, two, lesson_texts = minimal_sentences()
    banks = {name: [Bank.read(t, trial.lex) for t in texts]
             for name, texts in (('one', one), ('two', two), ('lesson', lesson_texts),
                                 ('held', HELD_OUT))}
    kw = dict(way=way, match=match)

    def measure():
        e = dict(Ep=Ep.detach(), Em=Em.detach(), **kw)
        return {name: count(trial, reader, banks[name], **e) for name in ('two', 'lesson', 'held')}

    record = dict(seed=seed, before=measure(), first_grad=None, curve=[])
    A0 = reader.A.detach().clone()
    for stage, (name, steps) in enumerate(zip(('one', 'two', 'lesson'), stages), 1):
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            loss = sum(read(trial, reader, bank, train=True, Ep=Ep, Em=Em, **kw)[0]
                       for bank in banks[name])
            loss.backward()
            if record['first_grad'] is None and name != 'one':
                record['first_grad'] = dict(
                    request=float(reader.A.grad.norm()) if reader.A.grad is not None else 0.,
                    entries=float(torch.cat([Ep.grad, Em.grad]).norm()) if Ep.grad is not None else 0.,
                    where=float(torch.cat([Ep.grad[:, :N_WHERE], Em.grad[:, :N_WHERE]]).norm())
                    if Ep.grad is not None else 0.,
                    when=float(torch.cat([Ep.grad[:, N_WHERE:], Em.grad[:, N_WHERE:]]).norm())
                    if Ep.grad is not None else 0.)
            optimizer.step()
            with torch.no_grad():
                Ep.clamp_(min=0.)
                Em.clamp_(min=0.)
            if step == steps:
                record['curve'].append((name, measure()['lesson']))
    e = dict(Ep=Ep.detach(), Em=Em.detach(), **kw)
    record['after'] = measure()
    record['match_off'] = count(trial, reader, banks['lesson'], Ep=Ep.detach(), Em=Em.detach(),
                                way=way, match=False)
    record['still'] = count(trial, reader, banks['lesson'], still=True, **e)
    norms = (Ep.detach() + Em.detach())
    where_norm = norms[:, :N_WHERE].sum(-1)
    top = where_norm.argsort(descending=True)[:5]
    record['entries_where'] = [(trial.lex.names[int(r)], round(float(where_norm[r]), 2)) for r in top]
    record['entries_when'] = round(float(norms[:, N_WHERE:].sum()), 3)
    record['request_moved'] = round(float((reader.A.detach() - A0).norm()), 3)
    # How much of the request reads the context of the words read (not the bias).
    record['request_context'] = round(float((reader.A.detach() - A0)[:, 1:].norm()), 3)
    record['trial'], record['reader'], record['Ep'], record['Em'] = trial, reader, Ep.detach(), Em.detach()
    return record


def own_place_shares(*, k=1., places=(0, 5, 10, 15, 20, 25), forward=True):
    """Measure (i): a row whose where entries are peaked at one place (``E+ =
    k c``, ``E- = k (1 - c)`` at the place's code), present at a token at
    each of several places; the where share at each token, three ways."""
    trial = PiWhere(lexicon(), forward=forward)
    lex = trial.lex
    text = ' '.join(['Felix'] * len(places))
    bank = Bank.read(text, lex)
    starts = [lo for lo, _ in bank.spans]
    r = lex.index['Felix']
    rows = {}
    for centre in (starts[0], starts[2], starts[4]):
        Ep, Em = trial.Ep.clone(), trial.Em.clone()
        code = trial.theta.detach()[centre]
        Ep[r, :N_WHERE], Em[r, :N_WHERE] = k * code, k * (1 - code)
        request = Request(words=0., concepts=torch.nn.functional.one_hot(torch.tensor(r), lex.R).to(DTYPE))
        for way in ('combined', 'lanes', 'entries'):
            _, s, _, _ = trial.shares(bank, bank.live(), request, way=way, Ep=Ep, Em=Em)
            rows[(centre, way)] = [round(float(v), 4) for v in s.detach()]
    return starts, rows


# --- A place as a row ---------------------------------------------------------
#
# A row of the first Pi with entries on the where wires and none on any what:
# it requires no content. A concept over it (``G`` on the row's value, as any
# row's concept) is a place; the request may ask for it beside "words" and the
# row concepts. The way down passes the row by its entries. The content rows
# keep no entry on a wire.

@dataclass
class PlaceRequest(Request):
    places: torch.Tensor = None             # [P], one for each place row


class PlaceTrial(PiWhere):
    """``PiWhere`` with ``P`` place rows: ``Pw`` [P, R] their what entries
    (none for a place), ``Pp``, ``Pm`` [P, D] their entries on the wires'
    two lanes. The content rows' entries on the wires stay at nothing."""

    def __init__(self, lex, Pw, Pp, Pm, *, labels=()):
        super().__init__(lex, forward=True, when=True)
        self.Pw, self.Pp, self.Pm = Pw.to(DTYPE), Pp.to(DTYPE), Pm.to(DTYPE)
        self.labels = tuple(labels)

    @property
    def P(self):
        return len(self.Pp)

    def up_whole(self, bank, activity, *, Ep=None, Em=None, wires=None, squash=True):
        out = super().up_whole(bank, activity, Ep=Ep, Em=Em, wires=wires, squash=squash)
        act = torch.tanh if squash else (lambda v: v)
        c = out['wires'].clamp(EPS, 1 - EPS)
        log_what = out['clamped'].log() @ self.Pw.T
        bp = (log_what + c.log() @ self.Pp.T + (1 - c).log() @ self.Pm.T).exp()      # [K, P]
        out.update(bp=bp, cwp=act(G * (1 - (-bp).exp())))
        return out

    def shares(self, bank, activity, request, *, way='entries', Ep=None, Em=None):
        what, s, when_share, out = super().shares(bank, activity, request, way=way, Ep=Ep, Em=Em)
        places = getattr(request, 'places', None)
        if places is None or not self.P:
            return what, s, when_share, out
        # The place concepts asked on their for lane (rule 36), each Sigma
        # passed by its weight G, each place row passed by its entries.
        bp = out['bp']
        gate = G * places.clamp_min(0.) * (-bp).exp() * bp                # [K, P]
        c = out['wires'].clamp(EPS, 1 - EPS)
        wire = c[:, None, :] * self.Pp + (1 - c)[:, None, :] * self.Pm    # [K, P, D]
        per_wire = (gate[..., None] * wire).sum(1)
        # A place row with what entries passes its share to them as the
        # Pi's transpose, as a content row does.
        d_m = (gate @ self.Pw) / out['clamped'] * (out['m'] >= EPS)
        at_max = (activity == out['m'][bank.token, bank.row]).to(activity.dtype)
        what = what + activity * d_m[bank.token, bank.row] * at_max
        return what, s + per_wire[:, :N_WHERE].sum(-1), when_share + per_wire[:, N_WHERE:].sum(-1), out


class PlaceReader(Reader):
    """The request over "words", the row concepts and the place concepts."""

    def __init__(self, trial, generator, noise=1e-2):
        R, P = trial.lex.R, trial.P
        self.R = R
        self.A = noise * torch.randn(R + P + 1, CONTEXT, generator=generator, dtype=DTYPE)
        self.A[0, 0] = 1.
        self.A.requires_grad_()

    def request(self, ctx, A=None):
        m = (self.A if A is None else A) @ ctx
        return PlaceRequest(words=m[0], concepts=m[1:self.R + 1], places=m[self.R + 1:])


def lesson_banks(lex):
    one, two, lesson_texts = minimal_sentences()
    return {name: [Bank.read(t, lex) for t in texts]
            for name, texts in (('one', one), ('two', two), ('lesson', lesson_texts), ('held', HELD_OUT))}


def word_starts(banks):
    """Every place a word starts at in the given sentences."""
    return sorted({bank.spans[k][0] for bank in banks for k in bank.word_tokens()})


def given_places(trial, places, *, when=False):
    """Rows of places given: entries at each place's code on the where wires,
    and with ``when`` at the when code on the when wires; none on a what."""
    P, D = len(places), trial.D
    Pp, Pm = torch.zeros(P, D, dtype=DTYPE), torch.zeros(P, D, dtype=DTYPE)
    code = trial.theta.detach()[torch.tensor(places, dtype=torch.long)]
    Pp[:, :N_WHERE], Pm[:, :N_WHERE] = code, 1 - code
    if when and trial.when.numel():
        Pp[:, N_WHERE:], Pm[:, N_WHERE:] = trial.when, 1 - trial.when
    return torch.zeros(P, trial.lex.R, dtype=DTYPE), Pp, Pm


def occurrences(trial, banks):
    """Each word token met: its start, and its coordinates in perceptual space
    (the what memberships of every row, and the wires' two lanes)."""
    met = []
    for bank in banks:
        out = PiWhere.up_whole(trial, bank, bank.live())
        what = (out['m'] >= .5).to(DTYPE)
        c = out['wires'].detach()
        for k in bank.word_tokens():
            met.append((bank.spans[k][0], bank.words[k], what[k], c[k], 1 - c[k]))
    return met


def cone_places(trial, banks, *, when=False):
    """Rows nominated by the cone's rule (rule 72): at each place, the least of
    its occurrences' coordinates: weight on what they share and none on what
    varies. A place met by one form only keeps that form's whats."""
    by_place = {}
    for start, form, what, cp, cm in occurrences(trial, banks):
        by_place.setdefault(start, []).append((form, what, cp, cm))
    rows, record = [], []
    for start in sorted(by_place):
        met = by_place[start]
        Pw = torch.stack([w for _, w, _, _ in met]).amin(0)
        Pp = torch.stack([p for _, _, p, _ in met]).amin(0)
        Pm = torch.stack([m for _, _, _, m in met]).amin(0)
        if not when:
            Pp[N_WHERE:], Pm[N_WHERE:] = 0., 0.
        rows.append((Pw, Pp, Pm))
        record.append(dict(place=start, occurrences=len(met), forms=sorted({f for f, *_ in met}),
                           whats=[trial.lex.names[i] for i in Pw.nonzero().flatten().tolist()]))
    Pw, Pp, Pm = (torch.stack(t) for t in zip(*rows))
    return (Pw, Pp, Pm), record


def candidate_offers(trial, banks):
    """What a Pi's pair candidates offer from the where wires: each pair of
    the wires' lanes tallied at every word token met. Returns the pairs and
    their worth, ``2 * least - 1`` of the two ratios of what the pair met to
    each lane's own presence; the tally has no place in it."""
    lanes = 2 * N_WHERE
    tallies, worth = {}, {}
    for start, _, _, cp, cm in occurrences(trial, banks):
        v = torch.cat([cp[:N_WHERE], cm[:N_WHERE]])
        for i in range(lanes):
            for j in range(i + 1, lanes):
                meet, left, right = tallies.get((i, j), (0., 0., 0.))
                tallies[i, j] = meet, left, right = (meet + min(float(v[i]), float(v[j])),
                                                     left + float(v[i]), right + float(v[j]))
                worth[(i, j)] = (-1. if left <= 0. or right <= 0. else
                                 2. * min(meet / left, meet / right) - 1.)
    return worth


def place_count(trial, reader, banks, **kw):
    with torch.no_grad():
        return sum(read(trial, reader, bank, **kw)[1] for bank in banks)


def teach_places(rows, *, seed=None, stages=(10, 10, 60), rate=.05, labels=()):
    """One start of the reading lesson with the place rows ``rows`` (fixed);
    Adam over the request's map alone."""
    seed = fresh_seed() if seed is None else seed
    generator = torch.Generator().manual_seed(seed)
    lex = lexicon()
    trial = PlaceTrial(lex, *rows, labels=labels)
    reader = PlaceReader(trial, generator)
    optimizer = torch.optim.Adam([reader.A], lr=rate)
    banks = lesson_banks(lex)

    def measure():
        return {name: place_count(trial, reader, banks[name]) for name in ('two', 'lesson', 'held')}

    record = dict(seed=seed, P=trial.P, before=measure())
    A0 = reader.A.detach().clone()
    for name, steps in zip(('one', 'two', 'lesson'), stages):
        for _ in range(steps):
            optimizer.zero_grad()
            loss = sum(read(trial, reader, bank, train=True)[0] for bank in banks[name])
            loss.backward()
            optimizer.step()
    record['after'] = measure()
    record['still'] = place_count(trial, reader, banks['lesson'], still=True)
    moved = reader.A.detach() - A0
    record['places_asked'] = round(float(moved[lex.R + 1:].norm()), 3)
    record['request_context'] = round(float(moved[:, 1:].norm()), 3)
    return record


def place_forms(when):
    lex = lexicon()
    trial = PiWhere(lex)
    banks = lesson_banks(lex)
    taught = banks['one'] + banks['two'] + banks['lesson']
    starts = word_starts(taught)
    given = given_places(trial, starts, when=when)
    nominated, record = cone_places(trial, taught, when=when)
    return starts, given, nominated, record


def place_report(runs=3):
    lex = lexicon()
    trial = PiWhere(lex)
    banks = lesson_banks(lex)
    taught = banks['one'] + banks['two'] + banks['lesson']
    starts, _, nominated, record = place_forms(False)
    print('Places a word starts at in the lesson:', starts)
    print('The cone over the occurrences at each place:')
    for r in record:
        print(f"  place {r['place']}: {r['occurrences']} occurrences of {r['forms']}; whats kept {r['whats']}")
    print('  rows nominated:', len(record), 'with no what:', sum(not r['whats'] for r in record))
    worth = candidate_offers(trial, taught)
    print('Pair candidates over the where lanes: pairs', len(worth), 'worth from',
          round(min(worth.values()), 3), 'to', round(max(worth.values()), 3), '(one tally for every place)')
    forms = []
    for when in (False, True):
        _, given, nominated, _ = place_forms(when)
        forms.append((f'places given, when {"in" if when else "out"}', given))
        forms.append((f'places nominated by the cone, when {"in" if when else "out"}', nominated))
    forms.append(('no place rows (the request alone)', given_places(trial, [])))
    for name, rows in forms:
        print(name)
        for _ in range(runs):
            r = teach_places(rows)
            print(f"  seed {r['seed']}: rows {r['P']}, before {r['before']}, after {r['after']}, "
                  f"request held at its start context {r['still']}, places asked (moved) "
                  f"{r['places_asked']}, request's context part moved {r['request_context']}")


def report(runs=3, **kw):
    for forward in (True, False):
      starts, rows = own_place_shares(forward=forward)
      print('Measure (i): where share of a row peaked at a place, wires in the forward', forward,
            'at tokens starting', starts)
      for (centre, way), values in rows.items():
        print(f'  peaked at {centre:2d}, {way:8s}:', values,
              'largest at own place' if starts[max(range(len(values)), key=values.__getitem__)] == centre
              else 'largest elsewhere')
    for variant, opts in (('as specified: where and when wires in the forward, by entries', {}),
                          ('as specified, rate .01', dict(rate=.01)),
                          ('where wires only', dict(when=False)),
                          ('where wires only, rate .01', dict(when=False, rate=.01)),
                          ('where wires only, out of the forward (the match alone)',
                           dict(when=False, forward=False)),
                          ('where wires only, rate .01, by the derivative combined (round 2)',
                           dict(when=False, rate=.01, way='combined')),
                          ('the request alone, no entries', dict(learn_entries=False))):
        print(variant)
        for _ in range(runs):
            r = teach(**opts, **kw)
            print(f"  seed {r['seed']}: before {r['before']}, after {r['after']}, curve {r['curve']}, "
                  f"match off {r['match_off']}, request held at its start context {r['still']}, "
                  f"first gradient {r['first_grad']}, request moved {r['request_moved']} "
                  f"(context {r['request_context']}), where entries {r['entries_where']}, "
                  f"when entries {r['entries_when']}")


if __name__ == '__main__':
    import sys
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    if len(sys.argv) > 2 and sys.argv[2] == 'places':
        place_report(runs)
    else:
        report(runs=runs)
