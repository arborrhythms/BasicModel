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
# Step 7: the toy of §19.126.

class Toy:
    """Places with learned codes ``codes`` ``[P, D]``; a request is a vector
    ``[D]`` read against them, so each place's activity is its code times
    the request. Two concepts with nothing learned at the start, each a
    learned read of the input (``entries`` ``[D, D]`` and ``bias`` ``[D]``,
    both at nothing): "first", lit while no word has been found, and
    "next", lit after. The lit concept's read of the input is the request of
    the next look. The input holds the last look's request: form ``A``, as
    given; form ``B``, as answered (the code of the place of the word the
    look returned). "first" has no last look: its input is nothing, so its
    request is its bias."""
    def __init__(self, places, width, form):
        self.form = form
        self.codes = torch.randn(places, width, requires_grad=True)
        self.entries = {name: torch.zeros(width, width, requires_grad=True) for name in ('first', 'next')}
        self.bias = {name: torch.zeros(width, requires_grad=True) for name in ('first', 'next')}

    def parameters(self):
        return [self.codes, *self.entries.values(), *self.bias.values()]

    def request(self, lit, given):
        """The lit concept's request ``[B, D]`` from the input ``given``."""
        return given @ self.entries[lit] + self.bias[lit]

    def read(self, sentences, *, forced=None, free_after=0):
        """A reading of each sentence (``[B, P]`` its live places), look by
        look: the request, each unread live place's activity, the place
        taken (the most active, a draw among equals; or ``forced`` ``[B,
        looks]`` where given, for the looks before ``free_after``). Returns
        the activities ``[B, looks, P]`` (``-inf`` off the unread live
        places) and the places taken ``[B, looks]``."""
        B, P = sentences.shape
        looks = int(sentences.sum(-1).max())
        unread = sentences.clone()
        given = self.codes.new_zeros(B, self.codes.shape[1])
        scores, taken = [], []
        for t in range(looks):
            asked = self.request('first' if t == 0 else 'next', given)
            activity = asked @ self.codes.t()
            activity = torch.where(unread, activity, -torch.inf)
            best = activity.detach().amax(-1, keepdim=True)
            tied = unread & (activity.detach() == best)
            place = torch.rand(B, P).masked_fill(~tied, -1.).argmax(-1)
            if forced is not None and t < free_after:
                place = forced[:, t]
            live = unread.any(-1)
            place = torch.where(live, place, -1)
            scores.append(activity)
            taken.append(place)
            unread = unread & ~((torch.arange(P)[None] == place[:, None]) & live[:, None])
            answered = self.codes[place.clamp_min(0)] * live[:, None]
            given = asked if self.form == 'A' else answered
        return torch.stack(scores, 1), torch.stack(taken, 1)


def toy_sentences(P, lengths, starts):
    rows = [(s, n) for n in lengths for s in starts if s + n <= P]
    live = torch.zeros(len(rows), P, dtype=torch.bool)
    for i, (s, n) in enumerate(rows):
        live[i, s:s + n] = True
    return live, rows


def toy_teach(toy, sentences, steps=600, rate=.05):
    """Reading targets: at each look the next place of the sentence, read in
    order (the reading supplied while it is taught), against the unread
    places; the loss before and after."""
    optimizer = torch.optim.Adam(toy.parameters(), lr=rate)
    P = sentences.shape[1]
    order = torch.where(sentences, torch.arange(P)[None], P).sort(-1).values
    looks = int(sentences.sum(-1).max())
    target = torch.where(order[:, :looks] < P, order[:, :looks], -1)
    first = last = None
    for _ in range(steps):
        optimizer.zero_grad()
        scores, _ = toy.read(sentences, forced=target.clamp_min(0), free_after=looks)
        live = target >= 0
        logp = scores.log_softmax(-1).gather(-1, target.clamp_min(0)[..., None])[..., 0]
        loss = -(torch.where(live, logp, 0.).sum() / live.sum())
        loss.backward()
        optimizer.step()
        last = float(loss.detach())
        first = last if first is None else first
    return first, last


def toy_measure(toy, sentences):
    """With no reading supplied: the first word right and the sentences read
    in order."""
    with torch.no_grad():
        _, taken = toy.read(sentences)
    P = sentences.shape[1]
    order = torch.where(sentences, torch.arange(P)[None], P).sort(-1).values[:, :taken.shape[1]]
    order = torch.where(order < P, order, -1)
    first = int((taken[:, 0] == order[:, 0]).sum())
    whole = int((taken == order).all(-1).sum())
    return first, whole, len(sentences)


def toy_pairs(toy, P, starts):
    """Codex's paired histories, in a sentence of five places at each start:
    the 2nd then the 4th supplied, the third look free (the 5th is next to
    the right); and the 4th then the 2nd (the 3rd). A pair counts if both
    free choices are right."""
    rows = [s for s in starts if s + 5 <= P]
    live = torch.zeros(len(rows), P, dtype=torch.bool)
    for i, s in enumerate(rows):
        live[i, s:s + 5] = True
    s = torch.tensor(rows)
    right = torch.ones(len(rows), dtype=torch.bool)
    for history, wanted in (((1, 3), 4), ((3, 1), 2)):
        forced = torch.stack((s + history[0], s + history[1], s), -1)
        with torch.no_grad():
            _, taken = toy.read(live, forced=forced, free_after=2)
        right &= taken[:, 2] == s + wanted
    return int(right.sum()), len(rows)


def toy_successor(toy, taught):
    """What "next" holds after learning: for each place p, the place its
    request (form B: from p's code) asks most among all other places, and
    among the places to its right. A successor for every place asks for p+1
    at every p; a table asks for it only at the places it was taught."""
    with torch.no_grad():
        P = len(toy.codes)
        asked = toy.request('next', toy.codes) @ toy.codes.t()                  # [P, P]
        asked = asked.masked_fill(torch.eye(P, dtype=torch.bool), -torch.inf)
        right = asked.masked_fill(torch.arange(P)[None] <= torch.arange(P)[:, None], -torch.inf)
        rows = range(P - 1)
        any_side = [p for p in rows if int(asked[p].argmax()) == p + 1]
        to_right = [p for p in rows if int(right[p].argmax()) == p + 1]
        inside = [p for p in rows if p in taught and p + 1 in taught]
    return dict(successor_among_all=f'{len(any_side)} of {P - 1}',
                successor_among_all_taught_places=f'{sum(p in inside for p in any_side)} of {len(inside)}',
                successor_to_the_right=f'{len(to_right)} of {P - 1}',
                successor_to_the_right_taught_places=f'{sum(p in inside for p in to_right)} of {len(inside)}',
                places_wrong_among_all=[p for p in rows if p not in any_side])


def test_the_first_and_next_concepts_a_toy():
    """Step 7 (§19.126): a toy of "first" and "next". 32 places; taught on
    sentences of 2 to 5 words starting at places 0 to 11, the reading
    supplied; measured with no reading supplied on those, on lengths 6 to 8
    and on starts 12 to 24 (not taught), and on the paired histories at
    every start. Both forms of the input, by drawn start. Asserted only that
    every count was made."""
    P, D = 32, 16
    taught_starts, untaught_starts = range(0, 12), range(12, 25)
    taught_lengths, untaught_lengths = (2, 3, 4, 5), (6, 7, 8)
    sets = dict(taught=toy_sentences(P, taught_lengths, taught_starts)[0],
                untaught_lengths=toy_sentences(P, untaught_lengths, taught_starts)[0],
                untaught_starts=toy_sentences(P, taught_lengths, untaught_starts)[0],
                both_untaught=toy_sentences(P, untaught_lengths, untaught_starts)[0])
    taught_places = set(range(0, 11 + 5))
    records = []
    for _ in range(runs()):
        start = fresh_start()
        for form in ('A', 'B'):
            toy = Toy(P, D, form)
            before = {name: toy_measure(toy, live) for name, live in sets.items()}
            loss = toy_teach(toy, sets['taught'])
            after = {name: toy_measure(toy, live) for name, live in sets.items()}
            pairs = dict(taught_starts=toy_pairs(toy, P, taught_starts),
                         untaught_starts=toy_pairs(toy, P, untaught_starts))
            records.append(dict(start=start, form=form, loss=tuple(round(v, 4) for v in loss),
                                before={k: f'first {f}, in order {w} of {n}' for k, (f, w, n) in before.items()},
                                after={k: f'first {f}, in order {w} of {n}' for k, (f, w, n) in after.items()},
                                pairs={k: f'{r} of {n}' for k, (r, n) in pairs.items()},
                                next_holds=toy_successor(toy, taught_places)))
            print(records[-1])
    assert len(records) == 2 * runs()


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
