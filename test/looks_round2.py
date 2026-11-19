"""A trial of the looks, in tests only (sigma-pi §22.2, round 2 of item 1.7).

Nothing here is wired into the model and nothing of ``bin/`` is changed;
``Spaces.WhereEncoding`` gives the constant sinusoid of a place.

- *The code of a form* (rule 43). Its characters, the space among them, and
  the runs added where two forms of the vocabulary would otherwise have the
  same parts: the shortest run that one holds and the other does not,
  carried by every form that holds it. No end marks and no lengths.
- *The bank* (rule 33). For each token, its form's row where the form is
  known and the rows of its parts, each at its place; a known form that
  spans several tokens is a token of its own at its start. Each event's band
  is its place's row of one parameter that holds the sinusoid (rule 47); the
  parameter takes a gradient and is given to no optimizer.
- *The way up.* The first Pi (rule 2): a form's row reads its own row and
  its parts, ``b = exp(W log m)``. The combination (rule 7), for each token,
  ``x_i = 1 - prod_p (1 - v_{p,i}) ** b_p``, over one what coordinate for
  each row and the four where coordinates. The Sigma (rule 3): one concept
  for each row (rule 21), ``tanh(G x_own + x_where . U_where)``, its where
  weights at zero at the start. The node "words" (rule 35) reads every word
  concept at ``G`` and has a Not entry on the space's concept.
- *The salience map* (rule 36). The request's gradient by each event's
  activity (the what) and by each event's band (the where), from the open
  run, with each Sigma passed before its squash; by the transposes with the
  local derivatives kept, and by the framework's backward.
- *The look* (rules 34, 37, 39, 47). What match, the share capped at one;
  where match, one of the forms tried; the way up again; ``interpret`` takes
  the most active word concept, and of forms that begin at one place the
  largest.
"""
from dataclasses import dataclass, field
import math

import torch

from looks_prototype import LESSON, TokenMax, tokens_of, where_code

DTYPE = torch.float64
EPS = 1e-6                 # the Pi's membership floor
G = 3.                     # a Sigma's weight on a row at its start
NOT_SPACE = 2 * G          # the Not entry of "words" on the space's concept
CAPACITY = 4096            # the registry capacity the prototype used (§20.6)
V_WHAT = 1 - math.exp(-1)  # a row's value on its own what coordinate

THREE = ('Fe li x is here.', 'xileF is here.', 'The cat is there.')
SENTENCES = tuple(LESSON) + THREE
ICE = 'This is ice cream.'
EXTRA_FORMS = ('act', 'ice', 'cream', 'ice cream')


def words_of(texts):
    return sorted({t[lo:hi] for t in texts for lo, hi in tokens_of(t) if t[lo:hi].isalpha()})


# The fifteen words of §19.52: the lesson's thirteen and `The`, `cat`.
WORDS = tuple(words_of(tuple(LESSON) + ('The cat is there.',)))


# --- The code of a form (rule 43) -------------------------------------------

def runs(form, length):
    return [form[i:i + length] for i in range(len(form) - length + 1)]


def shortest_difference(a, b):
    """The shortest run of characters that one form holds and the other not."""
    for length in range(2, max(len(a), len(b)) + 1):
        for one, other in ((a, b), (b, a)):
            for run in runs(one, length):
                if run not in other:
                    return run
    raise ValueError(f'{a!r} and {b!r} hold the same runs')


def part_map(forms):
    """The parts added, and each form's code: its characters and the added
    parts that lie inside it. Forms are admitted in order."""
    added = []

    def code(form):
        return frozenset(set(form) | {p for p in added if p in form})

    held = []
    for form in forms:
        for other in held:
            while form != other and code(form) == code(other):
                added.append(shortest_difference(form, other))
        held.append(form)
    return tuple(added), {form: code(form) for form in forms}


class Lexicon:
    """The rows: characters, added parts and forms; the first Pi's weights."""

    def __init__(self, forms, texts=()):
        self.forms = tuple(forms)
        self.added, self.codes = part_map(self.forms)
        chars = sorted({c for t in tuple(texts) + self.forms for c in t})
        names = list(chars) + [p for p in self.added if p not in chars]
        names += [f for f in self.forms if f not in names]
        self.names = tuple(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        W = torch.eye(len(names), dtype=DTYPE)
        for f in self.forms:
            for p in self.codes[f]:
                W[self.index[f], self.index[p]] = 1.
        self.W = W
        self.word_rows = torch.tensor([self.index[f] for f in self.forms])
        self.space = self.index.get(' ')
        self.part_names = tuple(chars) + tuple(p for p in self.added if p not in chars)

    @property
    def R(self):
        return len(self.names)


def lexicon(forms=WORDS + EXTRA_FORMS, texts=SENTENCES + (ICE,)):
    return Lexicon(forms, texts)


# --- The bank (rule 33) -----------------------------------------------------

@dataclass
class Bank:
    text: str
    spans: list              # tokens, (start, stop), by start
    form: list               # each token's form row, or None
    row: torch.Tensor        # [E] each event's row
    place: torch.Tensor      # [E] each event's place, its start
    token: torch.Tensor      # [E] each event's token
    head: torch.Tensor       # [K] the event at each token's start
    place_of: torch.Tensor   # [K, R] the place of each row's first event in a token

    @classmethod
    def read(cls, text, lex):
        units = tokens_of(text)
        spans = set(units)
        starts, stops = {lo for lo, _ in units}, {hi for _, hi in units}
        for f in lex.forms:
            if ' ' in f:
                at = text.find(f)
                while at >= 0:
                    if at in starts and at + len(f) in stops:
                        spans.add((at, at + len(f)))
                    at = text.find(f, at + 1)
        spans = sorted(spans)
        rows, places, tokens, heads, forms = [], [], [], [], []
        place_of = torch.zeros(len(spans), lex.R, dtype=torch.long)
        for k, (lo, hi) in enumerate(spans):
            s = text[lo:hi]
            place_of[k] = lo
            heads.append(len(rows))
            events = []
            if s in lex.codes:
                events.append((lex.index[s], lo))
                forms.append(lex.index[s])
            else:
                forms.append(None)
            for part in lex.part_names:
                for j in range(len(s) - len(part) + 1):
                    if s[j:j + len(part)] == part:
                        events.append((lex.index[part], lo + j))
            if s not in lex.codes:
                events.sort(key=lambda e: e[1])
            seen = set()
            for r, p in events:
                rows.append(r)
                places.append(p)
                tokens.append(k)
                if r not in seen:
                    place_of[k, r] = p
                    seen.add(r)
        return cls(text, spans, forms, torch.tensor(rows), torch.tensor(places),
                   torch.tensor(tokens), torch.tensor(heads), place_of)

    @property
    def words(self):
        return [self.text[lo:hi] for lo, hi in self.spans]

    @property
    def head_rows(self):
        return self.row[self.head]

    def live(self):
        return torch.ones(len(self.row), dtype=DTYPE)

    def word_tokens(self):
        """The tokens holding a known form, by start: what a reader reads."""
        return [k for k, f in enumerate(self.form) if f is not None]


# --- The where test concepts of step 5 --------------------------------------

@dataclass
class WhereConcept:
    """A concept given weight on the where by the test.

    ``kind='sigma'`` reads the where coordinates of the combined vector by
    the Sigma's weights, ``2k (c - 1/2)``: on a coordinate of value x the
    for lane at ``k c`` and the complement at ``k (1 - c)``, linearized.
    ``kind='pi'`` is a row of the Pi on the head event's where wires and on
    their complements, ``k c`` and ``k (1 - c)``, read by the Sigma at ``G``
    over its value at its own place (its magnitude).
    """
    kind: str
    centre: torch.Tensor
    k: float

    @property
    def weights(self):
        return 2 * self.k * (self.centre - .5)


# --- The way up -------------------------------------------------------------

class Trial:
    def __init__(self, lex, capacity=CAPACITY):
        self.lex, self.capacity = lex, capacity
        code = where_code(torch.arange(capacity), capacity).to(DTYPE)
        # The constant sinusoid on fixed wires (rule 47): a parameter, so a
        # gradient reaches it, and never given to an optimizer.
        self.theta = torch.nn.Parameter(code.clone())
        self.code = code
        self.Uw = torch.zeros(lex.R, 4, dtype=DTYPE)  # each concept's weights on the where
        # Which events' bands enter the combination's where coordinates:
        # 'combined', every event of a reading by its Pi value (rule 7 as
        # written); 'head', the event at the reading's start alone, by its
        # Pi value; 'start', that event's band as it is, whatever is present.
        self.where_of = 'combined'

    def where_weight(self, bank, b):
        """[K, R] the exponent of each row's band in the where coordinates."""
        K = len(bank.spans)
        if self.where_of == 'combined':
            return b
        mask = torch.zeros(K, self.lex.R, dtype=DTYPE)
        mask[torch.arange(K), bank.head_rows] = 1.
        return mask if self.where_of == 'start' else b * mask

    def band(self, bank):
        return self.theta[bank.place_of]                        # [K, R, 4]

    def up(self, bank, activity, *, band=None, squash=True, where=(), Uw=None):
        lex, K = self.lex, len(bank.spans)
        band = self.band(bank) if band is None else band
        Uw = self.Uw if Uw is None else Uw
        act = torch.tanh if squash else (lambda v: v)
        m = TokenMax.apply(activity, bank.token * lex.R + bank.row, K * lex.R).reshape(K, lex.R)
        clamped = m.clamp_min(EPS)
        b = (clamped.log() @ lex.W.T).exp()                     # the first Pi
        xw = 1 - (-b).exp()                                     # rule 7 on the what
        gap = (1 - band).clamp_min(EPS).log()
        E = (self.where_weight(bank, b)[..., None] * gap).sum(1)
        xp = 1 - E.exp()                                        # rule 7 on the where
        pre = G * xw + xp @ Uw.T
        cw = act(pre)
        pre_words = G * cw[:, lex.word_rows].sum(-1)
        if lex.space is not None:
            pre_words = pre_words - NOT_SPACE * cw[:, lex.space]
        words = act(pre_words)
        head_band = band[torch.arange(K), bank.head_rows]
        preL = []
        for spec in where:
            if spec.kind == 'sigma':
                preL.append(xp @ spec.weights)
            else:
                c, h = spec.centre.clamp(EPS, 1 - EPS), head_band.clamp(EPS, 1 - EPS)
                log = spec.k * (c * h.log() + (1 - c) * (1 - h).log()).sum(-1)
                peak = spec.k * (c * c.log() + (1 - c) * (1 - c).log()).sum(-1)
                preL.append(G * (log - peak).exp())
        preL = torch.stack(preL, -1) if preL else xp.new_zeros(K, 0)
        return dict(m=m, clamped=clamped, b=b, xw=xw, band=band, xp=xp, pre=pre, cw=cw,
                    pre_words=pre_words, words=words, preL=preL, L=act(preL), head_band=head_band)


@dataclass
class Request:
    """A mask over concepts: "words", the row concepts, and where concepts."""
    words: object = 0.
    concepts: torch.Tensor = None           # [R], for lane less against lane
    where: tuple = ()

    def value(self, out):
        total = (self.words * out['words']).sum()
        if self.concepts is not None:
            total = total + (out['cw'] @ self.concepts).sum()
        return total + out['L'].sum()

    @property
    def asks_what(self):
        words = self.words if not torch.is_tensor(self.words) else float(self.words.detach())
        return bool(words) or (self.concepts is not None and bool(self.concepts.ne(0).any()))


def concept_mask(lex, *forms):
    mask = torch.zeros(lex.R, dtype=DTYPE)
    for f in forms:
        mask[lex.index[f]] = 1.
    return mask


# --- The salience map (rule 36) ---------------------------------------------

def salience(trial, bank, activity, request, *, mode='weights', create_graph=False, Uw=None):
    """The what share of each event and the where share of each (token, row),
    by the framework's backward. ``mode='slope'`` passes each Sigma by its
    slope (the plain gradient); ``'weights'`` before its squash."""
    a = activity.detach().clone().requires_grad_()
    band = trial.band(bank)
    out = trial.up(bank, a, band=band, squash=(mode == 'slope'), where=request.where, Uw=Uw)
    ga, gb = torch.autograd.grad(request.value(out), (a, band), create_graph=create_graph,
                                 allow_unused=True)
    ga = torch.zeros_like(a) if ga is None else ga
    gb = torch.zeros_like(band) if gb is None else gb
    return activity * ga, (band * gb).sum(-1), out


def salience_by_transposes(trial, bank, activity, request, *, mode='weights'):
    """The same shares by the transposes of the way up's matrices, with the
    local derivatives kept from the open run. ``mode='entries'`` carries the
    request down by the rows' entries alone (no derivative), the what only."""
    lex, K, R = trial.lex, len(bank.spans), trial.lex.R
    out = trial.up(bank, activity, squash=(mode == 'slope'), where=request.where)
    slope = (lambda v: 1 - v ** 2) if mode == 'slope' else torch.ones_like
    d_words = request.words * slope(out['words'])
    d_cw = activity.new_zeros(K, R)
    if request.concepts is not None:
        d_cw = d_cw + request.concepts
    d_cw[:, lex.word_rows] += G * d_words[:, None]
    if lex.space is not None:
        d_cw[:, lex.space] -= NOT_SPACE * d_words
    d_pre = d_cw * slope(out['cw'])
    d_xw = G * d_pre
    tok, row = bank.token, bank.row
    if mode == 'entries':
        d_L = d_xw @ lex.W
        return activity * d_L[tok, row], activity.new_zeros(K, R), out
    b, band, xp = out['b'], out['band'], out['xp']
    d_xp = d_pre @ trial.Uw
    d_head = activity.new_zeros(K, 4)
    for j, spec in enumerate(request.where):
        d_preL = slope(out['L'][:, j])
        if spec.kind == 'sigma':
            d_xp = d_xp + d_preL[:, None] * spec.weights
        else:
            h = out['head_band']
            c = spec.centre.clamp(EPS, 1 - EPS)
            inside = ((h >= EPS) & (h <= 1 - EPS)).to(h.dtype)
            hc = h.clamp(EPS, 1 - EPS)
            d_head = d_head + (d_preL * out['preL'][:, j])[:, None] * spec.k * (c / hc - (1 - c) / (1 - hc)) * inside
    d_E = -d_xp * (1 - xp)                                       # x = 1 - exp(E)
    open_gap = ((1 - band) >= EPS).to(band.dtype)
    gap = (1 - band).clamp_min(EPS).log()
    weight = trial.where_weight(bank, b)
    d_weight = trial.where_weight(bank, torch.ones_like(b)) if trial.where_of != 'start' else 0.
    d_b = d_xw * (-b).exp() + (d_E[:, None, :] * gap).sum(-1) * d_weight
    d_band = -d_E[:, None, :] * weight[..., None] / (1 - band).clamp_min(EPS) * open_gap
    d_band[torch.arange(K), bank.head_rows] += d_head
    d_L = (d_b * b) @ lex.W                                      # the Pi's transpose
    d_m = d_L / out['clamped'] * (out['m'] >= EPS)
    at_max = (activity == out['m'][tok, row]).to(activity.dtype)
    return activity * d_m[tok, row] * at_max, (band * d_band).sum(-1), out


# --- The look ---------------------------------------------------------------

def where_match(share, request_forward=None, *, form='a'):
    """The where match of each token from its where share.

    (a) §21's form: the share referred to the most salient token and passed
        through the exponential, ``exp(s - max s)``.
    (b) the share capped, ``min(1, max(0, s))``.
    (c) the asked concepts' own forward reading of the where at each token,
        referred to the most active, ``exp(F - max F)``.
    ``'none'`` leaves every token at one (an ablation).
    """
    if form == 'a':
        return (share - share.max()).exp()
    if form == 'b':
        return share.clamp(0., 1.)
    if form == 'none':
        return torch.ones_like(share)
    return (request_forward - request_forward.max()).exp()


def look(trial, bank, request, *, activity=None, mode='weights', form='a', apply='once',
         create_graph=False, Uw=None, what_rule='times'):
    """One attend: the salience map onto the what and the where, the
    intersection, and the way up again. ``apply='once'`` puts the where
    match on a token's reading once (rule 39); ``'inside'`` on each event
    at its own place, before the products. ``what_rule='times'`` takes an
    event's activity in the look as its own times its share capped at one
    (rule 34); ``'min'`` as its share capped at its own activity."""
    lex, K = trial.lex, len(bank.spans)
    a0 = bank.live() if activity is None else activity
    Uw = trial.Uw if Uw is None else Uw
    if mode == 'entries':
        share, wshare, open_out = salience_by_transposes(trial, bank, a0, request, mode='entries')
    else:
        share, wshare, open_out = salience(trial, bank, a0, request, mode=mode,
                                           create_graph=create_graph, Uw=Uw)
    what = share.clamp(0., 1.) if request.asks_what else torch.ones_like(a0)
    if what_rule == 'min' and request.asks_what:
        # The share capped at the event's own activity, so that the
        # activity is not counted twice (the share already carries it).
        what = torch.where(a0 > 0, torch.minimum(a0, share.clamp_min(0.)) / a0.clamp_min(EPS), 0.)
    s = wshare[torch.arange(K), bank.head_rows]
    forward = None
    if form == 'c':
        forward = open_out['preL'].sum(-1)
        if request.concepts is not None:
            forward = forward + (open_out['xp'] @ Uw.T) @ request.concepts
    where = where_match(s, forward, form=form)
    if apply == 'inside':
        se = wshare[bank.token, bank.row]
        if form == 'a':
            we = (se - se.max()).exp()
        elif form == 'b':
            we = se.clamp(0., 1.)
        elif form == 'none':
            we = torch.ones_like(se)
        else:
            we = where[bank.token]
        out = trial.up(bank, a0 * what * we, Uw=Uw)
        word = out['cw'][:, lex.word_rows]
    else:
        out = trial.up(bank, a0 * what, Uw=Uw)
        word = out['cw'][:, lex.word_rows] * where[:, None]
    out.update(word=word, what=what, where=where, where_share=s, open=open_out)
    return out


@dataclass
class Taken:
    form: str = None
    token: int = None
    value: float = 0.
    tied: list = field(default_factory=list)

    @property
    def identified(self):
        return self.form is not None and not self.tied


def interpret(trial, bank, out, *, floor=1e-9, tol=1e-9):
    """The most active word concept, exactly. Of equals that begin at one
    place the largest is kept; equals left at different places are a tie
    (the request identified no word), reported in ``tied``."""
    word = out['word'].detach()
    best = float(word.max()) if word.numel() else 0.
    if best <= floor:
        return Taken()
    at = [tuple(p) for p in (word >= best - tol).nonzero().tolist()]
    forms = trial.lex.forms
    # Of equals that begin at one place, the largest (rule 37, §19.30).
    largest = {}
    for k, f in at:
        start = bank.spans[k][0]
        size = len(forms[f])
        if start not in largest or size > largest[start][0]:
            largest[start] = (size, [(k, f)])
        elif size == largest[start][0]:
            largest[start][1].append((k, f))
    at = [p for start in sorted(largest) for p in largest[start][1]]
    k, f = at[0]
    return Taken(forms[f], k, best, [(forms[j], i) for i, j in at[1:]])


def returned(trial, bank, out, floor=.01):
    """The returned set: (token, form) of every word concept at ``floor`` or more."""
    word = out['word'].detach()
    return {(k, trial.lex.forms[f]) for k, f in (word >= floor).nonzero().tolist()}


def present(trial, bank, asked):
    """The (token, form) pairs at which an asked form is the token's form."""
    return {(k, trial.lex.names[r]) for k, r in enumerate(bank.form)
            if r is not None and trial.lex.names[r] in asked}


# --- Reading in order (step 6) ---------------------------------------------

SLOTS = 4


def context(trial, bank, taken):
    """What has been read: for each slot of the taken words, the where code
    of its start, that it is occupied, and its age; and a bias."""
    values = [torch.ones(1, dtype=DTYPE)]
    for j in range(SLOTS):
        if j < len(taken):
            start = bank.spans[taken[j]][0]
            values += [trial.code[start], torch.tensor([1., (len(taken) - j) / SLOTS], dtype=DTYPE)]
        else:
            values.append(torch.zeros(6, dtype=DTYPE))
    return torch.cat(values)


CONTEXT = 1 + 6 * SLOTS


def minimal_sentences():
    """One word, then two, then the lesson's (rule 38: minimal sentences first)."""
    one = tuple(f'{w}.' for w in WORDS)
    two = tuple(dict.fromkeys(' '.join(t.split()[:2]).rstrip('.?') + '.' for t in LESSON))
    return one, two, tuple(LESSON)


class Reader:
    """The request as a learned map from the context to a mask over the
    concepts ("words" and every row's concept), starting at "words" alone."""

    def __init__(self, trial, *, generator=None, noise=0.):
        R = trial.lex.R
        self.A = torch.zeros(R + 1, CONTEXT, dtype=DTYPE)
        if noise:
            self.A += noise * torch.randn(R + 1, CONTEXT, generator=generator, dtype=DTYPE)
        self.A[0, 0] = 1.
        self.A.requires_grad_()

    def request(self, ctx):
        m = self.A @ ctx
        return Request(words=m[0], concepts=m[1:])


def read_in_order(trial, reader, bank, *, train=False, sharpness=10., Uw=None, form='a'):
    """Read a sentence look by look; return the loss and whether every word
    was taken in order. A taken form is lowered over its extent (the done mark)."""
    order = bank.word_tokens()
    activity = bank.live()
    loss, ok, taken = torch.zeros((), dtype=DTYPE), True, []
    for t, target in enumerate(order):
        request = reader.request(context(trial, bank, taken))
        out = look(trial, bank, request, activity=activity, create_graph=train, Uw=Uw, form=form)
        score = out['word'].max(-1).values[order]
        loss = loss - torch.log_softmax(sharpness * score, 0)[t]
        got = interpret(trial, bank, out)
        ok = ok and got.identified and got.token == target
        taken.append(target)
        activity = activity * (bank.token != target).to(DTYPE)
    return loss, ok


def count_in_order(trial, reader, banks, Uw=None, form='a'):
    return sum(read_in_order(trial, reader, bank, Uw=Uw, form=form)[1] for bank in banks)


def teach_order(trial, *, stages=(10, 10, 60), rate=.05, generator=None, noise=1e-2,
                where_learned=True, every=10, form='a', reading=True, rebuild=False,
                theta_learned=False, match='product'):
    """Teach the request (and the where weights) to read in order.

    ``reading`` adds the reading loss, ``rebuild`` the reconstruction's
    error over both pathways (rule 49). Returns the curve [(stage, step, in
    order of the lesson's 16)], the reader, the learned where weights and
    the where parameter's values after. Adam over the request map, the
    concepts' where weights when ``where_learned``, and the where parameter
    only when ``theta_learned``.
    """
    reader = Reader(trial, generator=generator, noise=noise)
    Uw = torch.zeros_like(trial.Uw, requires_grad=where_learned)
    groups = [reader.A] + ([Uw] if where_learned else []) + ([trial.theta] if theta_learned else [])
    optimizer = torch.optim.Adam(groups, lr=rate)
    lesson = [Bank.read(t, trial.lex) for t in LESSON]
    curve = [('start', 0, count_in_order(trial, reader, lesson, Uw=Uw.detach(), form=form))]
    for stage, (texts, steps) in enumerate(zip(minimal_sentences(), stages), 1):
        banks = [Bank.read(t, trial.lex) for t in texts]
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            loss = torch.zeros((), dtype=DTYPE)
            for bank in banks:
                if reading:
                    loss = loss + read_in_order(trial, reader, bank, train=True, Uw=Uw, form=form)[0]
                if rebuild:
                    loss = loss + reconstruct(trial, bank, Uw, match=match)[0]
            loss.backward()
            optimizer.step()
            if step % every == 0 or step == steps:
                curve.append((stage, step, count_in_order(trial, reader, lesson, Uw=Uw.detach(),
                                                          form=form)))
    trial.theta.grad = None
    return curve, reader, Uw.detach()


# --- The reconstruction over both pathways (rule 49, step 6b) ---------------

def recreated_band(Uw_row):
    """The band a concept gives what it recreates: the place its where
    weights read best, the inverse of ``2k (c - 1/2)`` at ``k = 1``."""
    return (.5 + Uw_row / 2).clamp(EPS, 1 - EPS)


def shift_band(band, offset, trial):
    """The band of a place ``offset`` further on: a rotation of each rung
    of the sinusoid, on fixed wires."""
    from Spaces import WhereEncoding
    w = WhereEncoding(nWhere=4).set_capacity(trial.capacity)
    s = 2 * band - 1
    out = []
    for i, period in ((0, w.maxVal), (2, w.period_hf)):
        d = 2 * math.pi * offset / period
        sin, cos = s[..., i], s[..., i + 1]
        out += [sin * math.cos(d) + cos * math.sin(d), cos * math.cos(d) - sin * math.sin(d)]
    return ((torch.stack(out, -1) + 1) / 2).clamp(EPS, 1 - EPS)


def place_weights(band, codes, *, match='product', k=None, sharpness=200.):
    """A recreated band's weight at each place of the out buffer.

    ``'product'`` (or ``'product<k>'``): the place row of §20.5 centred on
    the band, at concentration k (200 by default), read at each place's code, ``exp(k sum b log c + (1 - b) log(1 - c) - peak)``.
    ``'softmax'``: the inner product of the centred band and code, made a
    distribution over the buffer's places.
    """
    c = codes.clamp(EPS, 1 - EPS)
    if match.startswith('product'):
        k = float(match[len('product'):] or 200.) if k is None else k
        log = k * (band * c.log() + (1 - band) * (1 - c).log()).sum(-1)
        peak = k * (band * band.log() + (1 - band) * (1 - band).log()).sum(-1)
        return (log - peak).exp()
    return torch.softmax(sharpness * 4 * ((band - .5) * (c - .5)).sum(-1), 0)


def reconstruct(trial, bank, Uw, *, match='product'):
    """Put the sentence back from its concepts, through what and where.

    Each token's own word concept recreates its form's characters in order
    (rule 45); each character's band is the concept's recreated band moved
    on by its offset; its weight at each place is its band's match there;
    what is put back at a place is the sum of the characters times their
    weights, times the concept's value. Returns the squared error against
    the sentence's bytes at their places, and the buffer.
    """
    lex = trial.lex
    out = trial.up(bank, bank.live(), Uw=Uw)
    n = len(bank.text)
    codes = trial.theta[:n]
    buffer = torch.zeros(n, lex.R, dtype=DTYPE)
    for k, f in enumerate(bank.form):
        if f is None:
            continue
        value = out['cw'][k, f]
        band = recreated_band(Uw[f])
        for j, ch in enumerate(lex.names[f]):
            weights = place_weights(shift_band(band, j, trial), codes, match=match)
            buffer = buffer + value * weights[:, None] * torch.nn.functional.one_hot(
                torch.tensor(lex.index[ch]), lex.R).to(DTYPE)
    target = torch.nn.functional.one_hot(torch.tensor([lex.index[c] for c in bank.text]), lex.R).to(DTYPE)
    return ((buffer - target) ** 2).sum(), buffer


def rebuilt_letters(trial, banks, Uw, *, match='product'):
    """Letters whose own character has the most weight at their place, of all letters."""
    right = total = 0
    for bank in banks:
        _, buffer = reconstruct(trial, bank, Uw.detach(), match=match)
        for p, c in enumerate(bank.text):
            if c.isalpha():
                total += 1
                right += bool(float(buffer[p].max()) > 0 and int(buffer[p].argmax()) == trial.lex.index[c])
    return right, total
