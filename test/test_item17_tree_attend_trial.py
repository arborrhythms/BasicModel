"""Round 6d of item 1.7, tenth pass (sigma-pi §19.125, §19.126; §22.6d): a
trial in tests only. Nothing here is wired into the model; every learned
read below is the trial's own and provisional. No form is given the order
of a sentence, a preference for earlier places or a first-found rule: an
order is learned or it is absent. A tie is a draw. Records, not repairs:
no assertion is made on the outcome of a draw or of learning."""
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


# ---------------------------------------------------------------------------
# Step 1: the record of each look, and what the tree holds at each look.

def hold_the_tree(monkeypatch, into):
    """At each call of ``AttentionContext.context`` (once a look, before its
    request): the STM it is given and the sentence's field
    (``SentenceField``), as the context reads them."""
    import AttentionContext
    original = AttentionContext.context
    def context(model, stm, field, surprise=None):
        into.append(dict(stm=tuple(t.detach().clone() if torch.is_tensor(t) else t for t in stm),
                         window=None if field is None else field.support_window.detach().clone(),
                         positions=None if field is None else field.positions.clone(),
                         placements=None if field is None else {k: v.detach().clone()
                                                                for k, v in field.placements().items()}))
        return original(model, stm, field, surprise)
    monkeypatch.setattr(AttentionContext, 'context', context)


def test_each_look_records_its_request_and_the_where_and_code_of_its_word(tmp_path, monkeypatch):
    """Step 1: each look's decision (``Looks.decisions``) keeps the request
    it was given and the ``.where`` and the code of the word it took, read
    from the sentence's bank at that word's reading (``Places.readings``).
    Printed, at each look: what the tree being built holds (the walk's
    ``accepted``; the STM the context reads: occupied slots, reference rows,
    ``.where`` of each slot (STM field 11); the field's ``support_window``
    and ``placements``)."""
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
        tree = []
        hold_the_tree(monkeypatch, tree)
        with torch.no_grad():
            read(model, text)
        model.End()
        path, bank = seen[0], banks[0]
        wires, _, code = readings(bank)
        V = int(model._concept_owner().nVectors)
        for t, decision in enumerate(path.decisions):
            index = int(decision['index'][0])
            held = tree[t] if t < len(tree) else None
            print(dict(look=t, index=index, interpreted=bool(decision['interpreted'][0]),
                       where=[round(v, 4) for v in decision['where'][0].tolist()],
                       code_columns=decision['code'][0].nonzero().flatten().tolist(),
                       request_rows_asked=decision['request'][:, 0, 0].nonzero().flatten().tolist(),
                       stm_occupied=None if held is None else held['stm'][0][0].ne(0).any(-1).int().tolist(),
                       stm_reference_rows=None if held is None else held['stm'][4][0].tolist(),
                       stm_where_wires=None if held is None else [[round(v, 4) for v in row]
                                                                  for row in held['stm'][11][0].tolist()],
                       window=None if held is None or held['window'] is None else held['window'][0].int().tolist(),
                       placements_where=None if held is None or held['placements'] is None
                       else held['placements']['where'][0].tolist()))
            assert decision['request'].shape == (V, 1, 2)
            if index >= 0:
                assert torch.equal(decision['where'][0], wires[0, index])
                assert torch.equal(decision['code'][0], code[0, index])
        print(dict(accepted=path.walk.accepted.int().tolist(), looks=len(path.decisions), contexts=len(tree),
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


# ---------------------------------------------------------------------------
# Steps 3 to 5: the start, the tree presented, and the measures.

#: The four reads of the record compared (step 4): (a) the last word's code
#: and ``.where``; (b) every word found, summed (no order); (c) every word at
#: its place in the tree being built (the field's ``support_window``, newest
#: slot first); and Codex's first scorer, the STM the context reads (each
#: slot's contents, ``.where`` (field 11), ``.when`` (field 12) and
#: occupancy) and no ``.where`` of a look.
FORMS = ('last', 'every', 'tree', 'stm')
#: Codex's paired histories: (A, B) repeated as "A B A B A", after a prefix
#: of 0, 1 or 2 words (the starting place varied).
PAIR_WORDS = (('Felix', 'is'), ('Who', 'There'), ('Ciao', 'this'), ('Here', 'is'))
PREFIXES = ('', 'Who ', 'That is ')


def centred(wires):
    """The ``.where`` wires about zero: each rung's sine and cosine."""
    return 2 * wires - 1


class Trial:
    """The trial's reads, held by the test and not by the model. At a look
    in ``force`` mode the word taken is the next in source order (or the
    ``script``'s), a reading given, and what the look offers is kept as a
    lesson example; in ``free`` mode each unread word the look found is
    scored by the read of the record, the scores against its ``.where``
    wires (a bilinear read: ``centred(.where) . (entries @ record +
    bias)``); the first look, with nothing found, by a learned ``start`` on
    the ``.where`` wires alone. The look's own request is "words" (the
    model's ``request_start``; the context's map is at nothing before the
    lesson)."""
    def __init__(self, model):
        self.model, self.mode, self.form, self.path, self.tree = model, None, None, None, None
        self.script = self.wanted = None
        self.examples, self.log = [], []
        self.start = None
        self.reads = {}

    def install(self, monkeypatch):
        import SentenceLooks
        from SentenceLooks import Looks
        trial, run, original = self, Looks.run, SentenceLooks.word_activity
        trees = []
        hold_the_tree(monkeypatch, trees)
        def observed(path, arrive, **kwargs):
            trial.path = path
            trees.clear()
            try:
                return run(path, arrive, **kwargs)
            finally:
                trial.last = path
                trial.path = None
        def word_activity(field, concepts, known, extent, own, bank):
            words = original(field, concepts, known, extent, own, bank)
            if trial.mode is None or trial.path is None:
                return words
            trial.tree = trees[-1] if trees else None
            return trial.at_look(words, bank)
        monkeypatch.setattr(Looks, 'run', observed)
        monkeypatch.setattr(SentenceLooks, 'word_activity', word_activity)

    def found(self):
        return [(int(d['index'][0]), d['code'][0].float(), d['where'][0].float()) for d in self.path.decisions
                if bool(d['interpreted'][0]) and int(d['index'][0]) >= 0]

    def features(self, form, found):
        if form == 'last':
            _, code, where = found[-1]
            return torch.cat((code, centred(where)))
        if form == 'every':
            return torch.stack([torch.cat((code, centred(where))) for _, code, where in found]).sum(0)
        if form == 'tree':
            by = {w: torch.cat((code, centred(where))) for w, code, where in found}
            width = len(found[0][1]) + 4
            window, positions = self.tree['window'][0], self.tree['positions'].tolist()
            places = []
            for k in range(window.shape[0]):
                held = [by[positions[c]] for c in window[k].nonzero().flatten().tolist() if positions[c] in by]
                places.append(torch.stack(held).amax(0) if held else torch.zeros(width))
            return torch.cat(places)
        stm = self.tree['stm']
        contents = stm[0][0].float()
        occupied = contents.ne(0).any(-1, keepdim=True).float()
        return torch.cat((contents * occupied, stm[11][0].float(), stm[12][0].float(), occupied), -1).flatten()

    def query(self, form, x):
        entries, bias = self.reads[form]
        return entries @ x + bias

    def at_look(self, words, bank):
        from Places import readings
        state = self.path.walk
        W = words.shape[1]
        live = (state.spans[..., 1] > state.spans[..., 0])[0, :W]
        unread = live & ~state.accepted[0, :W]
        slots = [w for w in range(W) if bool(live[w])]
        t = len(self.path.decisions)
        candidates = [w for w in slots if bool(unread[w])]
        if self.wanted is not None and t == len(self.script):
            intended = slots[self.wanted]
        elif self.script is not None and t < len(self.script):
            intended = slots[self.script[t]]
        else:
            intended = candidates[0] if candidates else -1
        finite = intended >= 0 and bool(torch.isfinite(words[0, intended]))
        forced = self.mode == 'force' or (self.script is not None and t < len(self.script))
        self.log.append(dict(look=t, intended=intended, finite=finite, forced=forced))
        if self.mode == 'observe' or intended < 0 and forced:
            return words
        wires = readings(bank)[0][0, :W].float()
        found = self.found()
        if forced:
            if self.mode == 'force' and candidates:
                self.examples.append(dict(look=t, wires=wires[candidates], target=candidates.index(intended),
                                          features=None if not found else
                                          {form: self.features(form, found) for form in FORMS}))
            return torch.where(torch.arange(W) == intended, 1., -torch.inf)[None].to(words)
        with torch.no_grad():
            q = self.start if not found else self.query(self.form, self.features(self.form, found))
            scores = centred(wires) @ q
        return torch.where(torch.isfinite(words[0]) & unread, scores.to(words), -torch.inf)[None]

    def learn(self, steps=500, rate=.05):
        """Each read learned by its examples (the next word against the
        unread ones), from a start drawn afresh; the loss before and after
        and the share of examples whose target scores highest."""
        report = {}
        firsts = [e for e in self.examples if e['features'] is None]
        laters = [e for e in self.examples if e['features'] is not None]
        self.start = (.01 * torch.randn(4)).requires_grad_()
        jobs = [('start', [self.start], firsts, lambda e: self.start)]
        for form in FORMS:
            dim = len(laters[0]['features'][form])
            self.reads[form] = ((.01 * torch.randn(4, dim)).requires_grad_(), (.01 * torch.randn(4)).requires_grad_())
            jobs.append((form, list(self.reads[form]), laters,
                         lambda e, form=form: self.query(form, e['features'][form])))
        for name, parameters, examples, asked in jobs:
            optimizer = torch.optim.Adam(parameters, lr=rate)
            def loss():
                return torch.stack([-(centred(e['wires']) @ asked(e)).log_softmax(-1)[e['target']]
                                    for e in examples]).mean()
            before = float(loss().detach())
            for _ in range(steps):
                optimizer.zero_grad()
                value = loss()
                value.backward()
                optimizer.step()
            with torch.no_grad():
                right = sum(int((centred(e['wires']) @ asked(e)).argmax()) == e['target'] for e in examples)
                scale = max(float(asked(e).abs().max()) for e in examples)
            report[name] = dict(examples=len(examples), loss=(round(before, 4), round(float(value.detach()), 4)),
                                taught_right=right, query_greatest=round(scale, 3))
        for parameter in [self.start, *(p for r in self.reads.values() for p in r)]:
            parameter.requires_grad_(False)
        return report


def reading_of(model, trial, text, mode, *, form=None, script=None, wanted=None):
    from AttentionLesson import live_slots, looks_in_order
    from test_item17_places import staged
    from test_item18_expect import read
    trial.mode, trial.form, trial.script, trial.wanted = mode, form, script, wanted
    trial.log.clear()
    try:
        with torch.no_grad():
            read(model, text)
        model.End()
    finally:
        trial.mode = trial.script = trial.wanted = None
    path = trial.last
    slots = live_slots(path.walk.spans)[0]
    orders = path.source_orders()[0]
    forms = staged(model)
    return dict(in_order=all(looks_in_order(path)), slots=slots, orders=orders,
                first=bool(orders) and bool(slots) and orders[0] == slots[0],
                draws=int(sum(int(a.sum()) for a in path.walk.alternatives)),
                finite=[e['finite'] for e in trial.log if not e['forced']],
                taken=[int(d['index'][0]) for d in path.decisions],
                is_read=[w in orders for w in slots if forms[w] == 'is'])


def counts(readings):
    finite = [f for r in readings for f in r['finite']]
    is_read = [f for r in readings for f in r['is_read']]
    return dict(of=len(readings), in_order=sum(r['in_order'] for r in readings),
                first_word_first=sum(r['first'] for r in readings),
                intended_finite=f'{sum(finite)} of {len(finite)}', draws=sum(r['draws'] for r in readings),
                is_read=f'{sum(is_read)} of {len(is_read)}')


def pair_sentences():
    return [(prefix + ' '.join((a, b, a, b, a)) + '.', len(prefix.split()))
            for a, b in PAIR_WORDS for prefix in PREFIXES]


def the_request_at_the_first_two_looks(model, monkeypatch, forms):
    """§19.124, 1, as the ninth pass measured it: the query's greatest
    coordinate against the codes', and the gradient at the query and at the
    request, at the first two looks of "This is Felix." read with its
    reading named at rate nothing."""
    import SentenceLooks
    from AttentionLesson import reading_lesson
    from test_item18_expect import read
    asked, original = [], SentenceLooks.requested
    def requested(model, query):
        if query.requires_grad:
            query.retain_grad()
        request = original(model, query)
        if request.requires_grad:
            request.retain_grad()
        asked.append((query, request))
        return request
    monkeypatch.setattr(SentenceLooks, 'requested', requested)
    try:
        with reading_lesson(model, [forms['This is Felix.']]):
            read(model, 'This is Felix.', training=True, optimizer=model.getOptimizer(lr=0.))
        model.End()
    finally:
        monkeypatch.setattr(SentenceLooks, 'requested', original)
    codes = model._concept_owner().similarity_codebook.W.detach()
    grad = lambda t: None if t.grad is None else round(float(t.grad.abs().sum()), 6)
    return [dict(query_greatest=round(float(q.detach().abs().max()), 3),
                 codes_greatest=round(float(codes.abs().max()), 3),
                 query_gradient=grad(q), request_gradient=grad(r)) for q, r in asked[:2]]


def test_the_start_and_the_tree_presented_a_trial(tmp_path, monkeypatch):
    """Steps 3 to 5, recorded and not asserted. By start drawn afresh: the
    trial's reads learned from readings given in source order (the two-word
    and the lesson's sentences); then, with no reading supplied, each read
    on the two-word, the lesson's, the held-out and the reordered sentences
    and on Codex's paired histories. Then the reading lesson (``teach``) and
    the model's own reading, with ``request_start`` as built: the first word
    at the first look, the sentences in order, whether the intended word had
    a finite score at each choice, whether "is" is read, and the request and
    its gradient at the first two looks. Asserted only that every count was
    made."""
    from test_item17_order_trial import REORDERED
    from test_item17_places import HELD_OUT, minimal, model_for, nominate, staged_forms, teach
    one, two, lesson = minimal()
    sets = dict(two=two, lesson=LESSON_DISTINCT(), held=HELD_OUT, reordered=REORDERED)
    records = []
    for run in range(runs()):
        start = fresh_start()
        (tmp_path / str(run)).mkdir()
        model = model_for(tmp_path / str(run), monkeypatch)
        try:
            nominate(model, one + two + lesson + HELD_OUT + REORDERED)
            forms = {text: staged_forms(model, text) for text in one + two + lesson}
            trial = Trial(model)
            trial.install(monkeypatch)
            windows = []
            for text in two + LESSON_DISTINCT():
                reading_of(model, trial, text, 'force')
            merged = sum(int((e['features'] is not None) and False) for e in trial.examples)
            learned = trial.learn()
            by_form = {}
            for form in FORMS:
                found = {name: counts([reading_of(model, trial, text, 'free', form=form) for text in texts])
                         for name, texts in sets.items()}
                pairs = {}
                for text, k in pair_sentences():
                    both = []
                    for history, wanted in (((k + 1, k + 3), k + 4), ((k + 3, k + 1), k + 2)):
                        r = reading_of(model, trial, text, 'free', form=form, script=history, wanted=wanted)
                        both.append(len(r['taken']) > 2 and r['taken'][2] == r['slots'][wanted])
                    pairs.setdefault(k, []).append(all(both))
                found['pairs_by_prefix'] = {k: f'{sum(v)} of {len(v)}' for k, v in pairs.items()}
                by_form[form] = found
            # The model's own reading after the reading lesson.
            trial.mode = None
            teach(model, forms)
            own = {name: counts([reading_of(model, trial, text, 'observe') for text in texts])
                   for name, texts in sets.items()}
            gradient = the_request_at_the_first_two_looks(model, monkeypatch, forms)
            records.append(dict(start=start, examples=len(trial.examples), learned=learned, trial=by_form,
                                model_after_the_lesson=own, first_two_looks=gradient))
            print(records[-1])
        finally:
            model.End()
            monkeypatch.undo()
    assert len(records) == runs()


def LESSON_DISTINCT():
    from test_item17_places import LESSON
    return tuple(dict.fromkeys(LESSON))
