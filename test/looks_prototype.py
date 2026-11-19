"""Item 1.7 prototype of a look (sigma-pi §19.47): ask, pull back, go up.

Tests only. Nothing here is wired into the model and nothing of ``bin/``
is changed; ``WhereEncoding`` is read from ``bin/Spaces.py`` as the
constant sinusoid of a place.

The parts, as the record names them:

- *The bank* (rule 33). A sentence's percepts are read once: for each byte
  its row in the what codebook, its place and that place's sinusoid code,
  and the stem's token it belongs to. Activities are unsigned (the for
  lane); the against lane is empty in this prototype.
- *The first layer* (rule 2 of §19.21). Each row is a product over a
  token's memberships, ``z = exp(W log clamp(m, EPS, 1))``, with weights at
  zero or more: a word row has weight on its letters' codebook rows (the
  count of each), the fixed word whole has weight one on every byte of the
  token being a letter, and a place row has weight on the two unsigned
  lanes of the token's place code. The binding is by token (§19.45).
- *The Sigma* (rule 3). Each concept is ``tanh`` of a signed weight times
  its row; the meta concept "word" is one row over the word concepts.
- *The pullback* (rule 36, §19.43). A request, a mask over concepts, is
  carried down to each percept as its share, ``activity * dR/dactivity``.
  It is computed twice: by the transposes of the same matrices with the
  local derivatives kept from the way up, and by the framework's backward.
- *Two domains* (§19.48, answer 2). The request's rows on the what domain
  give each percept its what match (its share); its place rows, projected
  onto the where domain, give each percept its where match (the row read
  at the percept's own place). A look's activity is their intersection,
  here their product.
- *The returned set* (rule 37). The way up again on that activity: every
  concept at every token. ``interpret`` takes the most active word
  concept, exactly (§19.49); where two are equally active the one found
  first is taken (Claude's default there).
"""
from dataclasses import dataclass
import math
import re

import torch

EPS = 1e-6          # the membership floor of PiLayer2 (Layers.EPS_LOG)
GAIN = 3.           # the Sigma's weight on a row at full presence

LESSON = (
    'This is Felix.', 'What is this?', 'That is Ciao.', 'What is that?',
    'Here is Felix.', 'Who is here?', 'There is Ciao.', 'Who is there?',
    'That is Felix.', 'What is that?', 'This is Ciao.', 'What is this?',
    'Here is Ciao.', 'Who is here?', 'There is Felix.', 'Who is there?')


def tokens_of(text):
    """The stem's tokens: runs of letters, and each mark on its own."""
    return [(m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+|[^\sA-Za-z]", text)]


def where_code(places, capacity, rungs='both'):
    """The constant sinusoid of a place (``Spaces.WhereEncoding``)."""
    from Spaces import WhereEncoding
    code = WhereEncoding(nWhere=4).set_capacity(capacity).encode(
        torch.as_tensor(places)).to(torch.float64)
    if rungs == 'long':
        code = code[..., :2]
    elif rungs == 'short':
        code = code[..., 2:]
    return code


class TokenMax(torch.autograd.Function):
    """A token's membership of a codebook row: the greatest of its copies.

    The framework's own ``amax`` splits the gradient among tied copies, so
    a word with a repeated letter got half the share on each copy and came
    back at a power of a half. As a selection, every copy at the maximum is
    selected and takes the whole derivative.
    """
    @staticmethod
    def forward(ctx, activity, flat, size):
        m = torch.zeros(size, dtype=activity.dtype).scatter_reduce(
            0, flat, activity, 'amax', include_self=True)
        ctx.save_for_backward((activity == m[flat]).to(activity.dtype), flat)
        return m

    @staticmethod
    def backward(ctx, grad):
        at_max, flat = ctx.saved_tensors
        return grad[flat] * at_max, None, None


@dataclass
class Bank:
    """One sentence's percepts, read once."""
    text: str
    what: torch.Tensor      # [P] codebook row of each byte
    place: torch.Tensor     # [P] byte position
    token: torch.Tensor     # [P] token of each byte, -1 between tokens
    spans: list             # the stem's tokens, (start, stop)
    letter: torch.Tensor    # [P] the byte is a letter
    code: torch.Tensor      # [P, F] the sinusoid code of each byte's place
    start_code: torch.Tensor  # [K, F] the code of each token's first place

    @classmethod
    def read(cls, text, codebook, *, capacity=4096, rungs='both'):
        spans = tokens_of(text)
        token = torch.full((len(text),), -1, dtype=torch.long)
        for k, (lo, hi) in enumerate(spans):
            token[lo:hi] = k
        place = torch.arange(len(text))
        code = where_code(place, capacity, rungs)
        return cls(text, torch.tensor([codebook[c] for c in text]), place, token, spans,
                   torch.tensor([c.isalpha() for c in text], dtype=torch.float64),
                   code, code[torch.tensor([lo for lo, _ in spans])])

    @property
    def words(self):
        return [self.text[lo:hi] for lo, hi in self.spans]


class FirstLayer:
    """Word, whole and place rows over a token's memberships, and a Sigma."""

    def __init__(self, vocabulary, codebook):
        self.vocabulary, self.codebook = list(vocabulary), codebook
        W = torch.zeros(len(self.vocabulary), len(codebook), dtype=torch.float64)
        for row, word in enumerate(self.vocabulary):
            for c in word:
                W[row, codebook[c]] += 1.
        self.W = W          # [Nw, V], the count of each letter
        self.places = []    # (code, concentration), one place row each

    def add_place(self, code, concentration):
        """A row narrow on where: weights on both lanes of the place code.

        On a feature with code c the row weighs the for lane by kc and the
        complement lane by k(1-c), so its reading peaks at the place. Its
        radius narrows as k grows, and its value at the place, its
        magnitude, falls as exp(-k H); the Sigma's weight carries that.
        """
        self.places.append((code, float(concentration)))
        return len(self.places) - 1

    def place_log(self, code, index):
        """log z of a place row read at the given codes, beside its peak."""
        centre, k = self.places[index]
        c = code.clamp(EPS, 1 - EPS)
        log = (k * centre * c.log() + k * (1 - centre) * (1 - c).log()).sum(-1)
        cc = centre.clamp(EPS, 1 - EPS)
        peak = (k * centre * cc.log() + k * (1 - centre) * (1 - cc).log()).sum(-1)
        return log, peak

    # The way up, with the local derivatives kept for the pullback.
    def up(self, bank, activity, *, keep=False, sigma='slope'):
        """The way up. ``sigma='weights'`` reads each Sigma by its weights alone
        (its slope taken as one), which is how a pullback may pass it."""
        act = torch.tanh if sigma == 'slope' else (lambda value: value)
        K, V = len(bank.spans), len(self.codebook)
        inside = bank.token >= 0
        tok, what = bank.token[inside], bank.what[inside]
        a = activity[inside]
        flat = tok * V + what
        m = TokenMax.apply(a, flat, K * V)
        # Activities are at most one, so a membership is too. Only the floor
        # is clamped: the framework passes no gradient through a ceiling
        # reached exactly (PiLayer2 at full presence, torch 2.15), which is
        # where a word that is wholly present stands.
        clamped = m.clamp_min(EPS)
        L = clamped.log().reshape(K, V)
        zw = (L @ self.W.T).exp()                               # [K, Nw]
        x = (a * bank.letter[inside]).clamp_min(EPS)
        lw = torch.zeros(K, dtype=a.dtype).index_add(0, tok, x.log())
        zwh = lw.exp()                                          # [K]
        cw = act(GAIN * zw)
        cwh = act(GAIN * zwh)
        meta = act(GAIN * cw.sum(-1))
        if self.places:
            logs = [self.place_log(bank.start_code, i) for i in range(len(self.places))]
            # The Sigma's weight on a place row is its gain over its magnitude.
            cp = torch.stack([act(GAIN * (log - peak).exp()) for log, peak in logs], -1)
        else:
            cp = torch.zeros(K, 0, dtype=a.dtype)
        result = dict(word=cw, whole=cwh, meta=meta, place=cp, zw=zw)
        if keep:
            slope = (lambda c: 1 - c ** 2) if sigma == 'slope' else torch.ones_like
            result['kept'] = dict(inside=inside, flat=flat, m=m, clamped=clamped, zw=zw, x=x,
                                  zwh=zwh, a=a, tok=tok, d_cw=slope(cw), d_cwh=slope(cwh),
                                  d_meta=slope(meta))
        return result


@dataclass
class Request:
    """A mask over concepts on the for lane."""
    word: torch.Tensor = None       # [Nw]
    whole: float = 0.
    meta: float = 0.
    place: torch.Tensor = None      # [Np]

    def value(self, out):
        total = out['word'].new_zeros(())
        if self.word is not None:
            total = total + (out['word'] * self.word).sum()
        total = total + self.whole * out['whole'].sum() + self.meta * out['meta'].sum()
        if self.place is not None and out['place'].numel():
            total = total + (out['place'] * self.place).sum()
        return total

    @property
    def asks_what(self):
        return (self.word is not None and bool(self.word.any())) or self.whole or self.meta

    @property
    def asks_where(self):
        return self.place is not None and bool(self.place.any())


def pullback_by_transposes(layer, bank, activity, request, *, sigma='slope'):
    """Each percept's share, by the transposes of the way up's matrices."""
    out = layer.up(bank, activity, keep=True, sigma=sigma)
    kept = out['kept']
    K, V = len(bank.spans), len(layer.codebook)
    word = (torch.zeros_like(kept['zw']) if request.word is None else
            request.word.expand_as(kept['zw']).clone())
    # meta = act(G sum_w cw): its row carries down to every word concept.
    word = word + request.meta * kept['d_meta'][:, None] * GAIN
    d_zw = word * kept['d_cw'] * GAIN                          # [K, Nw]
    d_L = (d_zw * kept['zw']) @ layer.W                        # W transposed
    # The floor passes no gradient below it.
    d_m = d_L.reshape(-1) / kept['clamped'] * (kept['m'] >= EPS)
    # The max selects every copy at the maximum (TokenMax).
    a, flat = kept['a'], kept['flat']
    d_a = d_m[flat] * (a == kept['m'][flat]).to(a.dtype)
    # The whole: a product with weight one over the token's letters.
    d_zwh = request.whole * kept['d_cwh'] * GAIN
    x = kept['x']
    raw = kept['a'] * bank.letter[kept['inside']]
    d_a = d_a + (d_zwh * kept['zwh'])[kept['tok']] / x * (raw >= EPS) * bank.letter[kept['inside']]
    share = torch.zeros_like(activity)
    share[kept['inside']] = a * d_a
    return share


def pullback_by_backward(layer, bank, activity, request, *, sigma='slope'):
    """The same share by the framework's backward pass."""
    a = activity.detach().clone().requires_grad_()
    value = request.value(layer.up(bank, a, sigma=sigma))
    grad, = torch.autograd.grad(value, a, allow_unused=True)
    return torch.zeros_like(activity) if grad is None else a.detach() * grad


def projection(layer, bank, request):
    """The weights alone: each percept's letter read against the asked rows."""
    rows = torch.zeros(len(layer.vocabulary), dtype=torch.float64)
    if request.word is not None:
        rows = rows + request.word
    rows = rows + request.meta
    letters = rows @ (layer.W > 0).to(rows.dtype)              # [V]
    value = letters[bank.what] * (bank.token >= 0)
    if request.whole:
        value = value + request.whole * bank.letter * (bank.token >= 0)
    return value


def where_match(layer, bank, request, *, at='percept'):
    """The place rows asked for, read at each percept's own place.

    ``at='token'`` reads every percept at its token's first place instead,
    the binding by token applied to the where domain too.
    """
    if not request.asks_where:
        return torch.ones(len(bank.text), dtype=torch.float64)
    code = bank.code if at == 'percept' else bank.start_code[bank.token.clamp_min(0)]
    match = torch.zeros(len(bank.text), dtype=torch.float64)
    for index, weight in enumerate(request.place.tolist()):
        if weight:
            log, peak = layer.place_log(code, index)
            match = torch.maximum(match, weight * (log - peak).exp())
    return match * (bank.token >= 0)


def look(layer, bank, request, *, dim='inside', at='percept', sigma='weights'):
    """One attend: ask, pull back onto two domains, intersect, go up.

    ``dim='inside'`` applies the where match to each percept's activity
    before the products; ``dim='once'`` applies it once to a token's
    reading, after the way up. ``at`` reads the place rows at each
    percept's own place or at its token's first place. ``sigma`` passes
    the Sigma in the pullback by its slope (rule 36 as written) or by its
    weights alone.
    """
    activity = torch.ones(len(bank.text), dtype=torch.float64) * (bank.token >= 0)
    if request.asks_what:
        # The share is taken as the activity of the look, capped at full
        # presence: a letter of a wholly present asked row has a share of
        # at least one, and one shared by two asked rows is not counted twice.
        share = pullback_by_transposes(layer, bank, activity, request, sigma=sigma)
        what = share.clamp(0., 1.)
    else:
        what = activity
    where = where_match(layer, bank, request, at=at)
    if dim == 'inside':
        out = layer.up(bank, what * where)
    else:
        out = layer.up(bank, what)
        starts = torch.tensor([lo for lo, _ in bank.spans])
        factor = where[starts]
        out = dict(out, word=out['word'] * factor[:, None], whole=out['whole'] * factor,
                   meta=out['meta'] * factor)
    out['what'], out['where'], out['activity'] = what, where, what * where
    return out


def interpret(layer, bank, out, *, floor=1e-9, ties='first'):
    """The most active word concept of the returned set, exactly.

    Returns (word, token, value, tied): ``tied`` lists the other
    (word, token) pairs equally active. Nothing when no word concept is
    active. Among equals, ``ties='first'`` takes the one found first (the
    earliest token, then the vocabulary's order; Claude's default of
    §19.49) and ``ties='largest'`` the largest word (§19.30), by extent.
    """
    word = out['word']
    best = float(word.max()) if word.numel() else 0.
    if best <= floor:
        return None, None, 0., []
    at = [tuple(pair) for pair in (word == word.max()).nonzero().tolist()]
    if ties == 'largest':
        at.sort(key=lambda pair: (-len(layer.vocabulary[pair[1]]), pair[0]))
    k, w = at[0]
    return layer.vocabulary[w], k, best, [(layer.vocabulary[j], i) for i, j in at[1:]]


def lesson_layer(words=(), texts=()):
    """The lesson's words as rows; ``words`` adds known words, ``texts`` bytes."""
    every = list(LESSON) + list(words) + list(texts)
    codebook = {c: i for i, c in enumerate(sorted({c for t in every for c in t}))}
    vocabulary = sorted({t[lo:hi] for t in list(LESSON) + list(words)
                         for lo, hi in tokens_of(t) if t[lo:hi].isalpha()})
    return FirstLayer(vocabulary, codebook), codebook


def operations(layer, bank):
    """Multiply-adds of one way up and of one pullback, counted from shapes."""
    P = int((bank.token >= 0).sum())
    K, V, Nw = len(bank.spans), len(layer.codebook), len(layer.vocabulary)
    nnz = int((layer.W != 0).sum())
    up = P + K * V + K * nnz + K * Nw * 2 + P + K + K * Nw + K
    back = K * Nw * 3 + K * nnz + K * V * 2 + P * 3 + K + P * 3
    return up, back


def fallback(layer, bank, lo, hi):
    """§19.48's fallback: a .where given outright, then the largest word in it.

    "Largest" is read as the extent (§19.30 leaves it open). A word is
    taken only where its row is wholly present at a token inside the
    extent; failing a word, the largest concept there (the word whole).
    """
    activity = torch.ones(len(bank.text), dtype=torch.float64) * (bank.token >= 0)
    activity = activity * ((bank.place >= lo) & (bank.place < hi))
    out = layer.up(bank, activity)
    best = None
    for k, (a, b) in enumerate(bank.spans):
        if a < lo or b > hi:
            continue
        for w, word in enumerate(layer.vocabulary):
            if float(out['zw'][k, w]) >= 1 - 1e-9:
                key = (len(word), -k)
                if best is None or key > best[0]:
                    best = (key, 'word', word, k)
    if best is not None:
        return best[1:]
    wholes = [(b - a, k) for k, (a, b) in enumerate(bank.spans)
              if a >= lo and b <= hi and float(out['whole'][k]) > .5]
    if wholes:
        size, k = max(wholes)
        return 'whole', bank.text[slice(*bank.spans[k])], k
    return None, None, None


def place_profile(centre, offsets, *, capacity, rungs, concentration):
    """A place row's reading at other places, over its reading at its own."""
    places = torch.tensor([centre + d for d in offsets])
    code = where_code(places, capacity, rungs)
    middle = where_code(torch.tensor([centre]), capacity, rungs)[0]
    probe = FirstLayer([], {})
    probe.add_place(middle, concentration)
    log, peak = probe.place_log(code, 0)
    return (log - peak).exp()


def continuous_code(place, capacity):
    """``WhereEncoding.encode`` at a real-valued place, so a place can be taught."""
    from Spaces import WhereEncoding
    w = WhereEncoding(nWhere=4).set_capacity(capacity)
    shifted = place + w.where_origin
    long = 2 * math.pi * shifted / w.maxVal
    short = 2 * math.pi * shifted / w.period_hf
    return (torch.stack((long.sin(), long.cos(), short.sin(), short.cos()), -1) + 1) / 2


def order_trial(layer, codebook, *, start=-3., steps=150, rate=.05, capacity=4096,
                concentration=200., sharpness=10.):
    """Teach where the next look's place goes, with the gradient on the request alone.

    Each look asks for the meta concept and for a place at the end of the
    word read last plus a taught offset (the first look is at the start).
    The loss asks the next word of the sentence to be the most active. The
    offset starts inside the word just read. Returns the offset before and
    after, and how many sentences were read wholly in order each time.
    """
    offset = torch.tensor(float(start), dtype=torch.float64, requires_grad=True)
    banks = [Bank.read(text, codebook, capacity=capacity) for text in LESSON]
    request = Request(meta=1.)

    def run(train):
        total, in_order = offset.new_zeros(()), 0
        for bank in banks:
            live = (bank.token >= 0).to(torch.float64)
            what = pullback_by_transposes(layer, bank, live, request, sigma='weights').clamp(0, 1)
            words = [k for k, w in enumerate(bank.words) if w.isalpha()]
            ok = True
            for t, k in enumerate(words):
                place = (torch.zeros((), dtype=torch.float64) if t == 0 else
                         bank.spans[words[t - 1]][1] + offset)
                centre = continuous_code(place, capacity)
                c = bank.start_code.clamp(EPS, 1 - EPS)
                cc = centre.clamp(EPS, 1 - EPS)
                log = (concentration * (cc * c.log() + (1 - cc) * (1 - c).log())).sum(-1)
                peak = (concentration * (cc * cc.log() + (1 - cc) * (1 - cc).log())).sum(-1)
                rho = (log - peak).exp()
                out = layer.up(bank, what * rho[bank.token.clamp_min(0)] * live)
                score = out['word'].max(-1).values[words]
                total = total - torch.log_softmax(sharpness * score, 0)[t]
                taken, at, _, _ = interpret(layer, bank, dict(word=out['word'].detach()), ties='largest')
                ok &= at == k
            in_order += ok
        return total, in_order

    before = float(offset.detach())
    _, first = run(False)
    for _ in range(steps):
        loss, _ = run(True)
        grad, = torch.autograd.grad(loss, offset)
        with torch.no_grad():
            offset -= rate * grad
    _, last = run(False)
    return dict(offset_before=before, offset_after=float(offset.detach()), in_order_before=first,
                in_order_after=last, sentences=len(banks), steps=steps)
