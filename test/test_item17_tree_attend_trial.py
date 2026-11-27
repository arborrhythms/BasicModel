"""Round 6d of item 1.7, tenth pass (sigma-pi §19.125, §19.126; §22.6d): a
trial in tests only. Nothing here is wired into the model; every learned
read below is the trial's own and provisional. No form is given the order
of a sentence, a preference for earlier places or a first-found rule: an
order is learned or it is absent. A tie is a draw. Records, not repairs:
no assertion is made on the outcome of a draw or of learning. Steps 3 to 5
(the trial's reads of the record, the start, and the model's reading after
the reading lesson) were retired in the eleventh pass: the context's query,
the request's start and the reading lesson they read are gone (rule 80);
the model's own reading is measured in ``test_item17_own_reading``."""
import os

import torch
import torch.nn.functional as F


def fresh_start():
    """A start drawn afresh, printed by each caller."""
    start = int.from_bytes(os.urandom(6), 'little')
    torch.manual_seed(start)
    return start


def runs():
    return int(os.environ.get('TRIAL_RUNS', '3'))


# ---------------------------------------------------------------------------
# Step 7: the toy of §19.126, as Alec corrected it: the architecture as it
# stands, in small, with "first", "next" and a memory at the first Pi.

#: The toy's ``.where``: two rungs of the encoding's quadrature ladder, at
#: periods that resolve its 32 places (step 2: the model's 8192 and 256 do
#: not, at this scale).
TOY_PERIODS = (64, 16)


def toy_where(places):
    """Each place's ``.where`` wires ``[P, 4]``, as ``WhereEncoding`` makes
    them: ``(sin, cos + 1) / 2`` of each rung's phase."""
    a = torch.arange(places, dtype=torch.float64)
    rungs = [torch.stack(((2 * torch.pi * a / T).sin(), (2 * torch.pi * a / T).cos()), -1) for T in TOY_PERIODS]
    return ((torch.cat(rungs, -1) + 1) / 2).float()


class Toy:
    """The looks in small. A sentence is a run of percepts, one word each,
    each with its ``.what`` (its word's code) and its ``.where`` (its place's
    wires). The first Pi selects: a request ``[D]`` (on the ``.what`` and on
    the ``.where``, centred) carried down gives each unread percept its
    activity, the request read against its ``.what`` and ``.where``; the most
    active is selected, a draw among equals. Above it, the concepts a look
    lights: a word concept for each word (lit where its word is selected)
    and a place concept for each place (lit by its own ``.where`` against
    the selected one's, by shared size, as ``place_factor`` reads a row).
    Added, as Alec named them: two concepts with nothing learned at the
    start, "first", lit while nothing has been selected, and "next", lit
    after; the lit concept's read of the input is the next look's request.
    The input (``form``): ``B``, a memory at the first Pi of the ``.what``
    and the ``.where`` it selected at the previous look; ``C``, the previous
    look's conceptual field by wire (the activity of the concepts it lit),
    nothing kept at the first Pi; ``A`` (the comparison), the last request
    as given. "first" has no previous look: its input is nothing. The read
    (``read``): ``own``, a linear read of its own for each concept
    (``entries``, ``bias``, at nothing at the start); ``shared``, one small
    network over the input shared by both (``hidden`` units, rectified; its
    first layer drawn small, since at nothing it passes no gradient) with a
    linear read of its own for each concept on top (at nothing)."""
    HIDDEN = 16

    def __init__(self, places, vocabulary, width, form, read='own'):
        self.form, self.kind = form, read
        self.where = toy_where(places)
        self.what = F.normalize(torch.randn(vocabulary, width), dim=-1)
        D = width + self.where.shape[1]
        given = vocabulary + places if form == 'C' else D
        names = ('first', 'next')
        if read == 'own':
            self.entries = {n: torch.zeros(given, D, requires_grad=True) for n in names}
            self.bias = {n: torch.zeros(D, requires_grad=True) for n in names}
            self.shared = []
        else:
            H = self.HIDDEN
            self.shared = [(.1 * torch.randn(given, H)).requires_grad_(), torch.zeros(H, requires_grad=True)]
            self.entries = {n: torch.zeros(H, D, requires_grad=True) for n in names}
            self.bias = {n: torch.zeros(D, requires_grad=True) for n in names}

    def parameters(self):
        return [*self.shared, *self.entries.values(), *self.bias.values()]

    def request(self, lit, given):
        if self.kind == 'shared':
            given = torch.relu(given @ self.shared[0] + self.shared[1])
        return given @ self.entries[lit] + self.bias[lit]

    def table(self, words):
        """Each place's percept ``[B, P, D]``: its word's ``.what`` and its
        ``.where``, centred."""
        B, P = words.shape
        return torch.cat((self.what[words.clamp_min(0)], (2 * self.where - 1)[None].expand(B, -1, -1)), -1)

    def lit(self, words, place):
        """The conceptual field of a look that selected ``place`` ``[B]``:
        each word concept (one where its word was selected) and each place
        concept (``1 - mean |its .where - the selected .where|``)."""
        B = len(place)
        word = F.one_hot(words[torch.arange(B), place.clamp_min(0)].clamp_min(0), len(self.what)).float()
        at = self.where[place.clamp_min(0)]
        return torch.cat((word, 1 - (self.where[None] - at[:, None]).abs().mean(-1)), -1)

    def memory_at(self, place):
        """The input after a look that selected a percept at each place of
        ``place`` ``[n]``, its ``.what`` at nothing (B) or its word concept
        unlit (C)."""
        V = len(self.what)
        if self.form == 'C':
            field = self.lit(torch.zeros(len(place), len(self.where), dtype=torch.long), place)
            return torch.cat((torch.zeros(len(place), V), field[:, V:]), -1)
        return torch.cat((torch.zeros(len(place), self.what.shape[1]), 2 * self.where[place] - 1), -1)

    def read(self, words, *, forced=None, free_after=0):
        """Each sentence (``words`` ``[B, P]``, its word at each live place,
        -1 elsewhere) read look by look. Returns the activities ``[B, looks,
        P]`` (``-inf`` off the unread live places) and the places selected
        ``[B, looks]`` (``forced`` for the looks before ``free_after``)."""
        B, P = words.shape
        live = words >= 0
        percepts = self.table(words)
        looks = int(live.sum(-1).max())
        unread = live.clone()
        given = percepts.new_zeros(B, len(self.what) + P if self.form == 'C' else percepts.shape[-1])
        scores, taken = [], []
        for t in range(looks):
            asked = self.request('first' if t == 0 else 'next', given)
            activity = torch.where(unread, (percepts @ asked[..., None])[..., 0], -torch.inf)
            best = activity.detach().amax(-1, keepdim=True)
            tied = unread & (activity.detach() == best)
            place = torch.rand(B, P).masked_fill(~tied, -1.).argmax(-1)
            if forced is not None and t < free_after:
                place = forced[:, t]
            going = unread.any(-1)
            place = torch.where(going, place, -1)
            scores.append(activity)
            taken.append(place)
            unread = unread & ~((torch.arange(P)[None] == place[:, None]) & going[:, None])
            if self.form == 'B':
                # The memory at the first Pi: the .what and .where it selected.
                given = percepts[torch.arange(B), place.clamp_min(0)] * going[:, None]
            elif self.form == 'C':
                given = self.lit(words, place) * going[:, None]
            else:
                given = asked
        return torch.stack(scores, 1), torch.stack(taken, 1)


def toy_sentences(P, lengths, starts, vocabulary):
    """Sentences of each length at each start, each word drawn from the
    vocabulary, ``[N, P]``."""
    rows = [(s, n) for n in lengths for s in starts if s + n <= P]
    words = torch.full((len(rows), P), -1, dtype=torch.long)
    for i, (s, n) in enumerate(rows):
        words[i, s:s + n] = torch.randint(vocabulary, (n,))
    return words


def source_order(words):
    P = words.shape[1]
    order = torch.where(words >= 0, torch.arange(P)[None], P).sort(-1).values
    return torch.where(order < P, order, -1)


def toy_teach(toy, words, steps=600, rate=.05, drawn=False):
    """Reading targets: at each look the next place of the sentence (the
    reading supplied while it is taught) against the unread places; the
    loss before and after. ``drawn``: each sentence is also read once from
    a word drawn in it, rightwards to its end and then the words left of
    it in order, the cost taken at the looks after the first up to the
    end (the successor's examples; the words to the left stay unread
    rivals)."""
    optimizer = torch.optim.Adam(toy.parameters(), lr=rate)
    looks = int((words >= 0).sum(-1).max())
    order = source_order(words)[:, :looks]
    target, counted = order, order >= 0
    if drawn:
        n = (words >= 0).sum(-1)
        begin = (torch.rand(len(words)) * n).long()
        shifted, mask = torch.full_like(order, -1), torch.zeros_like(counted)
        for i in range(len(words)):
            row = order[i, :int(n[i])].tolist()
            b = int(begin[i])
            shifted[i, :len(row)] = torch.tensor(row[b:] + row[:b])
            mask[i, 1:len(row) - b] = True
        words = torch.cat((words, words))
        target, counted = torch.cat((order, shifted)), torch.cat((counted, mask))
    first = last = None
    for _ in range(steps):
        optimizer.zero_grad()
        scores, _ = toy.read(words, forced=target.clamp_min(0), free_after=looks)
        logp = scores.log_softmax(-1).gather(-1, target.clamp_min(0)[..., None])[..., 0]
        loss = -(torch.where(counted, logp, 0.).sum() / counted.sum())
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        first = last if first is None else first
    return first, last


def toy_measure(toy, words):
    """With no reading supplied: the first word right and the sentences read
    in order, ``(first, in order, of)``."""
    with torch.no_grad():
        _, taken = toy.read(words)
    order = source_order(words)[:, :taken.shape[1]]
    return int((taken[:, 0] == order[:, 0]).sum()), int((taken == order).all(-1).sum()), len(words)


def toy_pairs(toy, P, starts, vocabulary):
    """Codex's paired histories, "A B A B A" at each start (A and B drawn):
    the 2nd then the 4th supplied and the third look free (the 5th is next
    to the right), and the 4th then the 2nd (the 3rd). A pair counts if both
    free choices are right."""
    rows = [s for s in starts if s + 5 <= P]
    words = torch.full((len(rows), P), -1, dtype=torch.long)
    for i, s in enumerate(rows):
        a, b = torch.randperm(vocabulary)[:2].tolist()
        words[i, s:s + 5] = torch.tensor([a, b, a, b, a])
    s = torch.tensor(rows)
    right = torch.ones(len(rows), dtype=torch.bool)
    for history, wanted in (((1, 3), 4), ((3, 1), 2)):
        forced = torch.stack((s + history[0], s + history[1], s), -1)
        with torch.no_grad():
            _, taken = toy.read(words, forced=forced, free_after=2)
        right &= taken[:, 2] == s + wanted
    return int(right.sum()), len(rows)


def toy_successor(toy, taught):
    """What "next" holds after learning (forms B and C): with the input of a
    look that selected place p (its ``.what`` or word concept at nothing),
    the place whose ``.where`` the request asks most, among all others and
    among those to its right. One successor for every place asks for p+1 at
    every p; a table, only at the places it was taught."""
    if toy.form == 'A':
        return None
    with torch.no_grad():
        P = len(toy.where)
        W = toy.what.shape[1]
        asked = toy.request('next', toy.memory_at(torch.arange(P)))[:, W:] @ (2 * toy.where - 1).t()
        asked = asked.masked_fill(torch.eye(P, dtype=torch.bool), -torch.inf)
        right = asked.masked_fill(torch.arange(P)[None] <= torch.arange(P)[:, None], -torch.inf)
        rows = range(P - 1)
        any_side = [p for p in rows if int(asked[p].argmax()) == p + 1]
        to_right = [p for p in rows if int(right[p].argmax()) == p + 1]
        inside = [p for p in rows if p in taught and p + 1 in taught]
    return dict(all=f'{len(any_side)} of {P - 1}', all_taught=f'{sum(p in inside for p in any_side)} of {len(inside)}',
                right=f'{len(to_right)} of {P - 1}', right_taught=f'{sum(p in inside for p in to_right)} of {len(inside)}',
                wrong_among_all=[p for p in rows if p not in any_side])


def test_the_first_and_next_concepts_a_toy():
    """Step 7 (§19.126 with Alec's correction and his additions): a toy of
    the first Pi with "first" and "next". 32 places, six words; taught on
    sentences of 2 to 5 words starting at places 0 to 11, the reading
    supplied (and, in a second lesson, each sentence also read from a word
    drawn in it); measured with no reading supplied on those, on lengths 6
    to 8 and on starts 12 to 24 (not taught), and on the paired histories
    at every start. Inputs B (the memory at the first Pi), C (the last
    look's conceptual field by wire) and A (the request as given, the
    comparison); reads own and shared; by drawn start. Asserted only that
    every count was made."""
    P, vocabulary, width = 32, 6, 8
    taught_starts, untaught_starts = range(0, 12), range(12, 25)
    taught_lengths, untaught_lengths = (2, 3, 4, 5), (6, 7, 8)
    taught_places = set(range(0, 11 + 5))
    cases = [(form, read, drawn) for drawn in (False, True) for form in ('B', 'C') for read in ('own', 'shared')]
    cases.append(('A', 'own', False))
    records = []
    for _ in range(runs()):
        start = fresh_start()
        sets = dict(taught=toy_sentences(P, taught_lengths, taught_starts, vocabulary),
                    lengths=toy_sentences(P, untaught_lengths, taught_starts, vocabulary),
                    starts=toy_sentences(P, taught_lengths, untaught_starts, vocabulary),
                    both=toy_sentences(P, untaught_lengths, untaught_starts, vocabulary))
        for form, read, drawn in cases:
            toy = Toy(P, vocabulary, width, form, read)
            loss = toy_teach(toy, sets['taught'], drawn=drawn)
            after = {name: toy_measure(toy, words) for name, words in sets.items()}
            pairs = (toy_pairs(toy, P, taught_starts, vocabulary), toy_pairs(toy, P, untaught_starts, vocabulary))
            records.append(dict(start=start, form=form, read=read, drawn=drawn, loss=tuple(round(v, 3) for v in loss),
                                first={k: f'{f}/{n}' for k, (f, w, n) in after.items()},
                                in_order={k: f'{w}/{n}' for k, (f, w, n) in after.items()},
                                pairs=dict(taught_starts=f'{pairs[0][0]}/{pairs[0][1]}',
                                           untaught_starts=f'{pairs[1][0]}/{pairs[1][1]}'),
                                next_asks_p_plus_1=toy_successor(toy, taught_places)))
            print(records[-1])
    assert len(records) == len(cases) * runs()


# Step 8: the pairing of §19.127: the output "first, next*, end", no
# reading target.

def with_end(toy):
    """Gives ``toy`` an "end" concept with nothing learned: lit when it
    reads, by learned entries, the memory and whether the look found
    anything (``[memory, found]``); its read through a logistic is the
    chance that the sequence stops."""
    D = toy.what.shape[1] + toy.where.shape[1]
    given = len(toy.what) + len(toy.where) if toy.form == 'C' else D
    toy.end = (torch.zeros(given + 1, requires_grad=True), torch.zeros(1, requires_grad=True))
    return toy


def end_chance(toy, memory, found):
    entries, bias = toy.end
    return torch.sigmoid(torch.cat((memory, found[:, None].float()), -1) @ entries + bias)


def produce(toy, words):
    """The sentence produced: "first", then "next" at each later step, the
    lit concept's request selecting among the sentence's percepts (the
    most active unread one, a draw among equals; its word is the output),
    and at every step "end"'s chance of stopping. Returns, for each step,
    the output's word distribution (each percept's share of the softmax
    of the activities, summed over the percepts that hold a word), the
    chance of stopping, and the memory carried (from the percept selected:
    no gradient passes through the choice itself)."""
    B, P = words.shape
    V = len(toy.what)
    live = words >= 0
    percepts = toy.table(words)
    unread = live.clone()
    memory = percepts.new_zeros(B, V + P if toy.form == 'C' else percepts.shape[-1])
    outputs, stops = [], []
    holds = F.one_hot(words.clamp_min(0), V).float() * live[..., None]           # [B, P, V]
    for t in range(P + 1):
        found = unread.any(-1)
        stops.append(end_chance(toy, memory, found))
        asked = toy.request('first' if t == 0 else 'next', memory)
        activity = torch.where(unread, (percepts @ asked[..., None])[..., 0], -torch.inf)
        share = torch.where(found[:, None], torch.where(unread, activity, -1e9).softmax(-1), 0.)
        outputs.append((share[..., None] * holds).sum(1))
        best = activity.detach().amax(-1, keepdim=True)
        tied = unread & (activity.detach() == best)
        place = torch.where(found, torch.rand(B, P).masked_fill(~tied, -1.).argmax(-1), -1)
        unread = unread & ~((torch.arange(P)[None] == place[:, None]) & found[:, None])
        if toy.form == 'B':
            memory = percepts[torch.arange(B), place.clamp_min(0)] * found[:, None]
        elif toy.form == 'C':
            memory = toy.lit(words, place) * found[:, None]
        else:
            memory = asked
    return torch.stack(outputs, 1), torch.stack(stops, 1)


def toy_teach_output(toy, words, steps=600, rate=.05):
    """The output's cost alone: at each step the sentence's word in its
    order (the cross-entropy of the output's word distribution), and "end"
    at the step after the last word and not before. No reading target:
    which occurrence of a word is taken is never named. The loss before and
    after."""
    parameters = [*toy.parameters(), *toy.end]
    optimizer = torch.optim.Adam(parameters, lr=rate)
    order = source_order(words)
    n = (words >= 0).sum(-1)
    wanted = torch.where(order >= 0, words.gather(1, order.clamp_min(0)), -1)       # [B, P]
    first = last = None
    for _ in range(steps):
        optimizer.zero_grad()
        outputs, stops = produce(toy, words)
        steps_ = torch.arange(outputs.shape[1])[None]
        speaking = steps_ < n[:, None]
        target = F.pad(wanted, (0, 1), value=-1).clamp_min(0)
        logp = outputs.clamp_min(1e-9).log().gather(-1, target[..., None])[..., 0]
        word_cost = -(torch.where(speaking, logp, 0.).sum() / speaking.sum())
        ending = (steps_ == n[:, None]).float()
        counted = steps_ <= n[:, None]
        end_cost = torch.where(counted, F.binary_cross_entropy(stops, ending, reduction='none'), 0.).sum() / counted.sum()
        loss = word_cost + end_cost
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        first = last if first is None else first
    return first, last


def toy_reading_with_end(toy, words):
    """With no reading supplied: the reading by "first" and "next", stopped
    where "end" is more likely than not. Returns the first word right, the
    sentences read in order, and the sentences stopped exactly when every
    word was read (not before, and not after nothing was left)."""
    B, P = words.shape
    with torch.no_grad():
        live = words >= 0
        percepts = toy.table(words)
        unread = live.clone()
        memory = percepts.new_zeros(B, len(toy.what) + P if toy.form == 'C' else percepts.shape[-1])
        taken = torch.full((B, P), -1, dtype=torch.long)
        stopped = torch.full((B,), -1, dtype=torch.long)
        for t in range(P + 1):
            found = unread.any(-1)
            going = stopped < 0
            stop = going & (end_chance(toy, memory, found) > .5)
            stopped = torch.where(stop, t, stopped)
            going = going & ~stop & found
            if t == P or not bool(going.any()):
                break
            asked = toy.request('first' if t == 0 else 'next', memory)
            activity = torch.where(unread, (percepts @ asked[..., None])[..., 0], -torch.inf)
            tied = unread & (activity == activity.amax(-1, keepdim=True))
            place = torch.where(going, torch.rand(B, P).masked_fill(~tied, -1.).argmax(-1), -1)
            taken[:, t] = place
            unread = unread & ~((torch.arange(P)[None] == place[:, None]) & going[:, None])
            selected = percepts[torch.arange(B), place.clamp_min(0)] * going[:, None]
            fresh = selected if toy.form == 'B' else (toy.lit(words, place) * going[:, None] if toy.form == 'C' else asked)
            memory = torch.where(going[:, None], fresh, memory)
    order = source_order(words)
    n = live.sum(-1)
    first = int((taken[:, 0] == order[:, 0]).sum())
    whole = int(((taken == order) | (order < 0)).all(-1).sum())
    at_end = int((stopped == n).sum())
    early = int(((stopped >= 0) & (stopped < n)).sum())
    return first, whole, at_end, early, B


def test_first_next_end_on_the_output_a_toy():
    """Step 8 (§19.127): the toy of step 7 with no reading target. The cost
    is the output's alone: the sentence produced in its order by "first",
    then "next", then "end" (a third concept that stops the sequence); the
    reading uses the same "first" and "next" and is stopped by "end". Beside
    it, step 7's lesson (reading targets, "end" taught on the output as
    well, so that its reading stops by the same rule). Same sizes as step 7;
    measured with no reading supplied; by drawn start. Asserted only that
    every count was made."""
    P, vocabulary, width = 32, 6, 8
    taught_starts, untaught_starts = range(0, 12), range(12, 25)
    taught_lengths, untaught_lengths = (2, 3, 4, 5), (6, 7, 8)
    cases = [(form, read, 'output') for form in ('B', 'C') for read in ('own', 'shared')]
    cases += [('B', 'own', 'reading targets'), ('C', 'own', 'reading targets')]
    records = []
    for _ in range(runs()):
        start = fresh_start()
        sets = dict(taught=toy_sentences(P, taught_lengths, taught_starts, vocabulary),
                    lengths=toy_sentences(P, untaught_lengths, taught_starts, vocabulary),
                    starts=toy_sentences(P, taught_lengths, untaught_starts, vocabulary),
                    both=toy_sentences(P, untaught_lengths, untaught_starts, vocabulary))
        for form, read, lesson in cases:
            toy = with_end(Toy(P, vocabulary, width, form, read))
            if lesson == 'output':
                loss = toy_teach_output(toy, sets['taught'])
            else:
                loss = toy_teach(toy, sets['taught'])
                # "end" alone is then taught on the output, the rest held.
                for parameter in toy.parameters():
                    parameter.requires_grad_(False)
                toy_teach_output(toy, sets['taught'], steps=200)
            measured = {name: toy_reading_with_end(toy, words) for name, words in sets.items()}
            pairs = (toy_pairs(toy, P, taught_starts, vocabulary), toy_pairs(toy, P, untaught_starts, vocabulary))
            records.append(dict(start=start, form=form, read=read, lesson=lesson, loss=tuple(round(v, 3) for v in loss),
                                first={k: f'{m[0]}/{m[4]}' for k, m in measured.items()},
                                in_order={k: f'{m[1]}/{m[4]}' for k, m in measured.items()},
                                stopped_at_the_end={k: f'{m[2]}/{m[4]}' for k, m in measured.items()},
                                stopped_early={k: f'{m[3]}/{m[4]}' for k, m in measured.items()},
                                pairs=dict(taught_starts=f'{pairs[0][0]}/{pairs[0][1]}',
                                           untaught_starts=f'{pairs[1][0]}/{pairs[1][1]}'),
                                next_asks_p_plus_1=toy_successor(toy, set(range(0, 16)))))
            print(records[-1])
    assert len(records) == len(cases) * runs()


# Step 9: no concept told what it is (§19.128); reading and counting.

class Lit:
    """The toy of step 7 with nothing named. One network, shared by every
    task: its input is the concepts lit (one unused concept for each task,
    ``tasks``), the memory at the first Pi (the ``.what`` and ``.where`` it
    selected at the last look: nothing before the first look), whether the
    last look found anything (nothing left), and the last word of the
    output; ``hidden`` rectified units; three linear reads on top: the next
    request, the chance of stopping, and the output word over the output's
    vocabulary (the sentence's words, then the counting words, opaque). The
    output word is read from the same units given the percept the look
    selected (its share of the softmax of the activities, so the output's
    cost reaches the look's scores). The first look, later looks and the end
    are told apart only by what the memory holds."""
    HIDDEN = 32

    def __init__(self, places, objects, width, tasks=2, vocabulary=10):
        self.where = toy_where(places)
        self.what = F.normalize(torch.randn(objects, width), dim=-1)
        self.tasks, self.vocabulary = tasks, vocabulary
        D = width + self.where.shape[1]
        given = tasks + D + 1 + vocabulary
        H = self.HIDDEN
        self.first_layer = (.1 * torch.randn(given, H)).requires_grad_()
        self.first_bias = torch.zeros(H, requires_grad=True)
        self.second_layer = (.1 * torch.randn(given + D, H)).requires_grad_()
        self.second_bias = torch.zeros(H, requires_grad=True)
        self.ask = torch.zeros(H, D, requires_grad=True)
        self.stop = torch.zeros(H, requires_grad=True)
        self.say = torch.zeros(H, vocabulary, requires_grad=True)

    def parameters(self):
        return [self.first_layer, self.first_bias, self.second_layer, self.second_bias, self.ask, self.stop, self.say]

    def run(self, words, task, *, said=None, forced=None, free_after=0, steps=None):
        """A sequence over ``words`` ``[B, P]`` (an object or word index at
        each live place) with ``task``'s concept lit. ``said`` ``[B, T]``:
        the output's words to carry in the memory (in a lesson), else the
        sequence's own. Returns the output logits ``[B, T, V]``, the stop
        logits ``[B, T]`` (before each look) and the places taken."""
        B, P = words.shape
        live = words >= 0
        percepts = torch.cat((self.what[words.clamp_min(0)], (2 * self.where - 1)[None].expand(B, -1, -1)), -1)
        D = percepts.shape[-1]
        lit = F.one_hot(torch.full((B,), task), self.tasks).float()
        unread = live.clone()
        memory, last = percepts.new_zeros(B, D), percepts.new_zeros(B, self.vocabulary)
        steps = P if steps is None else steps
        outputs, stops, taken = [], [], []
        for t in range(steps):
            found = unread.any(-1)
            x = torch.cat((lit, memory, found[:, None].float(), last), -1)
            h = torch.relu(x @ self.first_layer + self.first_bias)
            stops.append(h @ self.stop)
            asked = h @ self.ask
            activity = torch.where(unread, (percepts @ asked[..., None])[..., 0], -torch.inf)
            share = torch.where(found[:, None], torch.where(unread, activity, -1e9).softmax(-1), 0.)
            seen = (share[..., None] * percepts).sum(1)
            g = torch.relu(torch.cat((x, seen), -1) @ self.second_layer + self.second_bias)
            logits = g @ self.say
            outputs.append(logits)
            tied = unread & (activity.detach() == activity.detach().amax(-1, keepdim=True))
            place = torch.where(found, torch.rand(B, P).masked_fill(~tied, -1.).argmax(-1), -1)
            if forced is not None and t < free_after:
                place = torch.where(found, forced[:, t], -1)
            taken.append(place)
            unread = unread & ~((torch.arange(P)[None] == place[:, None]) & found[:, None])
            memory = percepts[torch.arange(B), place.clamp_min(0)] * found[:, None]
            word = (said[:, t] if said is not None and t < said.shape[1] else logits.detach().argmax(-1))
            last = F.one_hot(word.clamp_min(0), self.vocabulary).float() * (word >= 0)[:, None]
        return torch.stack(outputs, 1), torch.stack(stops, 1), torch.stack(taken, 1)


#: The output's vocabulary in step 9: the six words (0 to 5), then the
#: counting words "1" to "4" (6 to 9).
COUNTING = 6


def reading_targets_of(words):
    """The reading task's output: the sentence's words in their order."""
    order = source_order(words)
    return torch.where(order >= 0, words.gather(1, order.clamp_min(0)), -1)


def counting_targets_of(objects):
    """The counting task's output: "1" to "4", one for each object (none
    beyond four)."""
    n = (objects >= 0).sum(-1)
    k = torch.arange(objects.shape[1])[None]
    return torch.where((k < n[:, None]) & (k < 4), COUNTING + k, -1)


def lit_teach(net, words, task, targets, steps=800, rate=.02, *, also=None):
    """The output's cost alone: each step's word, and stopping before the
    look after the last word and not before. ``also``: a second task's
    ``(words, task, targets)`` taught at the same steps (the two costs
    summed). The loss before and after."""
    optimizer = torch.optim.Adam(net.parameters(), lr=rate)
    first = last = None
    for _ in range(steps):
        optimizer.zero_grad()
        loss = lit_cost(net, words, task, targets)
        if also is not None:
            loss = loss + lit_cost(net, *also)
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        first = last if first is None else first
    return round(first, 3), round(last, 3)


def lit_cost(net, words, task, targets):
    n = (words >= 0).sum(-1)
    T = int(n.max()) + 1
    if True:
        outputs, stops, _ = net.run(words, task, said=targets[:, :T], steps=T)
        k = torch.arange(T)[None]
        speaking = (k < n[:, None]) & (F.pad(targets, (0, max(0, T - targets.shape[1])), value=-1)[:, :T] >= 0)
        target = F.pad(targets, (0, max(0, T - targets.shape[1])), value=-1)[:, :T].clamp_min(0)
        word_cost = F.cross_entropy(outputs.transpose(1, 2), target, reduction='none')
        word_cost = torch.where(speaking, word_cost, 0.).sum() / speaking.sum().clamp_min(1)
        counted = k <= n[:, None]
        stop_cost = F.binary_cross_entropy_with_logits(stops, (k == n[:, None]).float(), reduction='none')
        return word_cost + torch.where(counted, stop_cost, 0.).sum() / counted.sum()


def lit_measure(net, words, task, targets, *, forced=None, free_after=0):
    """With nothing supplied: the sequence until it stops (stop more likely
    than not before a look) or every place is spent. Returns, for each
    sequence, the places looked at before the stop, the output said, and
    the look at which it stopped (-1: never)."""
    with torch.no_grad():
        outputs, stops, taken = net.run(words, task, forced=forced, free_after=free_after)
    stop = (stops > 0)
    stopped = torch.where(stop.any(-1), stop.int().argmax(-1), -1)
    return taken, outputs.argmax(-1), stopped


def reading_counts(net, words):
    taken, said, stopped = lit_measure(net, words, 0, None)
    n = (words >= 0).sum(-1)
    order = source_order(words)
    looks = torch.arange(taken.shape[1])[None] < n[:, None]
    in_order = ((taken == order) | ~looks).all(-1)
    spoken = ((said == reading_targets_of(words)) | ~looks).all(-1)
    return dict(first=f'{int((taken[:, 0] == order[:, 0]).sum())}/{len(words)}',
                in_order=f'{int(in_order.sum())}/{len(words)}', said_right=f'{int(spoken.sum())}/{len(words)}',
                stopped_at_the_end=f'{int((stopped == n).sum())}/{len(words)}',
                stopped_early=f'{int(((stopped >= 0) & (stopped < n)).sum())}/{len(words)}')


def counting_counts(net, objects):
    """By number of objects: the counts said right (each step's counting
    word up to four, stopped after the last object), each object looked at
    once before the stop, and what was said."""
    taken, said, stopped = lit_measure(net, objects, 1, None)
    n = (objects >= 0).sum(-1)
    want = counting_targets_of(objects)
    by = {}
    for k in sorted(set(n.tolist())):
        rows = (n == k).nonzero().flatten()
        right = once = 0
        for b in rows.tolist():
            end = int(stopped[b]) if int(stopped[b]) >= 0 else taken.shape[1]
            looked = [p for p in taken[b, :end].tolist() if p >= 0]
            once += sorted(looked) == sorted((objects[b] >= 0).nonzero().flatten().tolist())
            right += end == k and all(int(said[b, i]) == int(want[b, i]) for i in range(min(k, 4)))
        sample = said[rows[0], :int(stopped[rows[0]]) if int(stopped[rows[0]]) >= 0 else taken.shape[1]].tolist()
        by[k] = dict(right=f'{right}/{len(rows)}', each_looked_at_once=f'{once}/{len(rows)}',
                     said=[f'"{v - COUNTING + 1}"' if v >= COUNTING else f'word {v}' for v in sample])
    return by


def lit_pairs(net, P, starts, vocabulary=6):
    rows = [s for s in starts if s + 5 <= P]
    words = torch.full((len(rows), P), -1, dtype=torch.long)
    for i, s in enumerate(rows):
        a, b = torch.randperm(vocabulary)[:2].tolist()
        words[i, s:s + 5] = torch.tensor([a, b, a, b, a])
    s = torch.tensor(rows)
    right = torch.ones(len(rows), dtype=torch.bool)
    for history, wanted in (((1, 3), 4), ((3, 1), 2)):
        forced = torch.stack((s + history[0], s + history[1], s), -1)
        taken, _, _ = lit_measure(net, words, 0, None, forced=forced, free_after=2)
        right &= taken[:, 2] == s + wanted
    return f'{int(right.sum())}/{len(rows)}'


def objects_at(P, counts, per_count):
    """Scenes of ``counts`` objects each at places drawn without
    replacement, the object's kind drawn of three, ``[N, P]``."""
    rows = []
    for k in counts:
        for _ in range(per_count):
            row = torch.full((P,), -1, dtype=torch.long)
            row[torch.randperm(P)[:k]] = torch.randint(3, (k,))
            rows.append(row)
    return torch.stack(rows)


def lit_shared(net):
    """What the two lit concepts share and what differs: their rows of the
    first layer (cosine), the hidden units each lights at the first look
    (memory nothing), and the place each asks most at the first look over
    all 32 places."""
    with torch.no_grad():
        rows = net.first_layer[:2]
        cosine = float(F.cosine_similarity(rows[0], rows[1], dim=0))
        units = []
        asked = []
        for task in range(2):
            x = torch.zeros(1, net.first_layer.shape[0])
            x[0, task] = 1.
            x[0, net.tasks + net.what.shape[1] + 4] = 1.          # found: something is there
            h = torch.relu(x @ net.first_layer + net.first_bias)[0]
            units.append(set((h > 0).nonzero().flatten().tolist()))
            q = h @ net.ask
            asked.append(int(((2 * net.where - 1) @ q[net.what.shape[1]:]).argmax()))
    return dict(first_layer_cosine=round(cosine, 3), units_lit_by_both=len(units[0] & units[1]),
                only_reading=len(units[0] - units[1]), only_counting=len(units[1] - units[0]),
                first_place_asked=dict(reading=asked[0], counting=asked[1]))


def test_one_lit_concept_for_each_task_reading_and_counting_a_toy():
    """Step 9 (§19.128), a toy recorded and not asserted. (a) Reading: one
    unused concept lit, the output's cost alone, nothing named; taught on
    sentences of 2 to 5 words at starts 0 to 11; measured on those, on
    untaught lengths and starts, and on the paired histories. (b) Counting:
    a second unused concept lit; scenes of 1 to 4 objects at drawn places
    (5 and 6 held out); the output "1" to "4", one per object. Taught in
    both orders (reading then counting, counting then reading), each task
    measured after each lesson; and both taught together. By drawn start; asserted only that every
    count was made."""
    P, width = 32, 8
    records = []
    for _ in range(runs()):
        start = fresh_start()
        reading = dict(taught=toy_sentences(P, (2, 3, 4, 5), range(0, 12), 6),
                       lengths=toy_sentences(P, (6, 7, 8), range(0, 12), 6),
                       starts=toy_sentences(P, (2, 3, 4, 5), range(12, 25), 6))
        scenes = objects_at(P, (1, 2, 3, 4), 12)
        held = objects_at(P, (1, 2, 3, 4, 5, 6), 6)
        for order in (('reading', 'counting'), ('counting', 'reading'), ('together',)):
            net = Lit(P, 6, width)
            stages = []
            for task in order:
                if task == 'together':
                    loss = lit_teach(net, reading['taught'], 0, reading_targets_of(reading['taught']),
                                     also=(scenes, 1, counting_targets_of(scenes)))
                elif task == 'reading':
                    loss = lit_teach(net, reading['taught'], 0, reading_targets_of(reading['taught']))
                else:
                    loss = lit_teach(net, scenes, 1, counting_targets_of(scenes))
                stages.append(dict(after=task, loss=loss,
                                   reading={k: reading_counts(net, w) for k, w in reading.items()},
                                   pairs=dict(taught_starts=lit_pairs(net, P, range(0, 12)),
                                              untaught_starts=lit_pairs(net, P, range(12, 25))),
                                   counting=counting_counts(net, held)))
            records.append(dict(start=start, order=order, stages=stages, shared=lit_shared(net)))
            print(records[-1])
    assert len(records) == 3 * runs()


# ---------------------------------------------------------------------------
# Step 1: the record of each look, and what the tree holds at each look.

def test_each_look_records_its_request_and_the_where_and_code_of_its_word(tmp_path, monkeypatch):
    """Step 1: each look's decision (``Looks.decisions``) keeps the request
    it was given and the ``.where`` and the code of the word it took, read
    from the sentence's bank at that word's reading (``Places.readings``).
    Printed, at each look: the word taken and the request. Ported (eleventh
    pass): the STM and the field the context read at each look are no
    longer read by ``attend`` (rule 80), so they are not printed here."""
    from Places import readings
    from test_item17_places import model_for, nominate, observe_runs
    from test_item18_expect import read
    model = model_for(tmp_path, monkeypatch)
    try:
        text = 'This is Felix.'
        nominate(model, (text,))
        seen = observe_runs(monkeypatch)
        banks = []
        from SentenceLooks import Looks
        run = Looks.run
        def kept(self, arrive, **kwargs):
            try:
                return run(self, arrive, **kwargs)
            finally:
                banks.append(model._sentence_bank)
        monkeypatch.setattr(Looks, 'run', kept)
        with torch.no_grad():
            read(model, text)
        model.End()
        path, bank = seen[0], banks[0]
        wires, _, code = readings(bank)
        V = int(model._concept_owner().nVectors)
        for t, decision in enumerate(path.decisions):
            index = int(decision['index'][0])
            print(dict(look=t, index=index, interpreted=bool(decision['interpreted'][0]),
                       where=[round(v, 4) for v in decision['where'][0].tolist()],
                       code_columns=decision['code'][0].nonzero().flatten().tolist(),
                       request_rows_asked=decision['request'][:, 0, 0].nonzero().flatten().tolist()))
            # After the concepts, the request for a place is a start and an
            # extent on the where wires (4d74726ce8), two of their width.
            assert decision['request'].shape == (V + 2 * wires.shape[-1], 1, 2)
            if index >= 0:
                assert torch.equal(decision['where'][0], wires[0, index])
                assert torch.equal(decision['code'][0], code[0, index])
        print(dict(accepted=path.walk.accepted.int().tolist(), looks=len(path.decisions),
                   interpret_bias=getattr(path.walk.chooser, 'interpret_bias', None)))
        assert path.decisions
    finally:
        model.End()


# ---------------------------------------------------------------------------
# Step 2: does a request for a place tell its word from the neighbours?

def per_word(values, token, W):
    """The greatest of ``values`` ``[O]`` over each word's occurrences, ``[W]``."""
    out = values.new_full((W,), -torch.inf)
    for w in range(W):
        held = token == w
        if bool(held.any()):
            out[w] = values[held].max()
    return out


def margin(values, p, slots):
    """``values`` at word ``p`` less the greatest at the other live words."""
    others = [float(values[q]) for q in slots if q != p and torch.isfinite(values[q])]
    if not torch.isfinite(values[p]) or not others:
        return None
    return float(values[p]) - max(others)


def single_period_codes(addresses, period):
    """One quadrature pair at ``period``, as the where encoding makes each
    rung: ``(sin, cos + 1) / 2`` of the address's phase; ``[n, 2]``."""
    phase = 2 * torch.pi * addresses.double() / period
    return ((torch.stack((phase.sin(), phase.cos()), -1) + 1) / 2).float()


def test_a_request_for_a_place_against_its_neighbours(tmp_path, monkeypatch):
    """Step 2, a record. On the lesson's distinct sentences, for every word
    p: a request for p's place concept alone, carried down as the source
    carries it (``Salience.salience``, ``SentenceLooks.look``), the share
    and the returned activity at p and at every other word; then a request
    on p's ``.where`` wires directly (shared size, ``field_factor``, as
    ``place_factor`` reads a row's entries). The margin between p and its
    nearest rival in each. Then the direct request's margin with the
    ``.where`` made of one quadrature pair at each of several periods
    (measured, not built). Asserted only that places were asked."""
    from AttentionLesson import live_slots
    from BasicPercepts import WHERE_WIRES, field_factor
    from Places import readings
    from Salience import salience
    from SentenceLooks import Looks, extent_of, look, word_activity
    from test_item17_places import minimal, model_for, nominate, places_of
    from test_item18_expect import read
    model = model_for(tmp_path, monkeypatch)
    try:
        one, two, lesson = minimal()
        nominate(model, one + two + lesson)
        basic, concept = places_of(model)
        by_key = {tuple(round(v, 6) for v in basic.place_rows[row][0].tolist()): inventory
                  for row, inventory in concept.items()}
        owner = model._concept_owner()
        V = int(owner.nVectors)
        rows, starts = [], []
        run = Looks.run
        def probe(self, arrive, **kwargs):
            bank = model.__dict__.get('_sentence_banks', {}).get(self.sid)
            if bank is not None and not rows_done[0]:
                rows_done[0] = True
                model._sentence_bank = bank
                W = self.walk.known.shape[1]
                slots = live_slots(self.walk.spans)[0]
                concepts = model._attention_word_concepts[:, :W]
                wires, _, _ = readings(bank)
                token = bank.token[0]
                own = bank.activity.clone()
                lanes = bank.table[..., bank.width:bank.width + WHERE_WIRES, :]
                starts.append([int(bank.where[0, p, 0]) for p in slots])
                for p in slots:
                    entries = torch.stack((wires[0, p], 1 - wires[0, p]), -1)[None].float()
                    direct = per_word(field_factor(lanes.float(), entries)[0, 0], token, W)
                    record = dict(text=text_now[0], p=p, direct=margin(direct, p, slots),
                                  direct_taken=int(direct.argmax()) == p)
                    inventory = by_key.get(tuple(round(v, 6) for v in wires[0, p].float().tolist()))
                    if inventory is not None:
                        request = owner.similarity_codebook.W.new_zeros(V, 1, 2)
                        request[inventory, 0, 0] = 1.
                        share = per_word(salience(owner, bank, request).events[0].sum(-1), token, W)
                        field, activity = look(model, bank, request, own)
                        returned = word_activity(field, concepts, self.walk.known,
                                                 extent_of(bank, activity, W), own, bank)[0]
                        record.update(share=margin(share, p, slots), share_taken=int(share.argmax()) == p,
                                      returned=margin(returned, p, slots),
                                      returned_taken=bool(torch.isfinite(returned[p]))
                                      and int(returned.argmax()) == p)
                    rows.append(record)
            return run(self, arrive, **kwargs)
        monkeypatch.setattr(Looks, 'run', probe)
        rows_done, text_now = [False], [None]
        texts = tuple(dict.fromkeys(lesson))
        with torch.no_grad():
            for text in texts:
                rows_done[0], text_now[0] = False, text
                read(model, text)
                model.End()
        for r in rows:
            print({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
        def summary(key):
            values = [r[key] for r in rows if r.get(key) is not None]
            return dict(words=len(values), taken=sum(r.get(key + '_taken', False) for r in rows),
                        least=round(min(values), 6) if values else None,
                        mean=round(sum(values) / len(values), 6) if values else None)
        encoding = model.where_registry.encoding
        long, short = encoding.maxVal, encoding.maxVal // encoding.rung_ratio
        periods = {}
        for period in (4, 8, 16, 32, 64, 128, 256, short, long):
            least, total, taken = None, 0., 0
            count = 0
            for addresses in starts:
                a = torch.tensor(addresses)
                codes = single_period_codes(a, period)
                lanes = torch.stack((codes, 1 - codes), -1)[None]                  # [1, n, 2, 2]
                for i in range(len(a)):
                    entries = lanes[0, i][None]
                    score = field_factor(lanes, entries)[0, 0]
                    m = margin(score, i, range(len(a)))
                    least = m if least is None else min(least, m)
                    total += m
                    taken += int(score.argmax()) == i
                    count += 1
            periods[period] = dict(least=round(least, 6), mean=round(total / count, 6), taken=f'{taken} of {count}')
        print(dict(places=len(concept), sentences=len(texts), starts=starts,
                   through_the_place_concept=dict(share=summary('share'), returned=summary('returned')),
                   on_the_where_wires=summary('direct'),
                   ladder=dict(long=long, short=short, text_slice=model.where_registry.slices['text']),
                   single_period=periods))
        assert any('share' in r for r in rows)
    finally:
        model.End()
