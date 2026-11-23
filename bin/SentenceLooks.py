"""A sentence's looks: each a request carried down onto the sentence's
percept bank, read up again, and what it returns handed to the grammar.

A look (§21, rules 34, 37, 39, 79 and 80): the request is ``attend``'s read
of the conceptual field (:mod:`RequestRead`); an occurrence's activity is
its salience, the request carried down by the ways back, capped at its own
activity; a reading's where match is ``exp(s - max s)`` of the salience of
what it holds, applied once to the reading's values; the way up again on a
table of the bank's most salient occurrences gives the returned set. The
most active word is taken (a draw among equals), and a word interpreted is
lowered in the bank's own activity over its extent.
"""
from dataclasses import fields
import torch
from torch.nn import functional as F

from RequestRead import at_rows, spread_once


def perceive(model, activity):
    """A look's read of the sentence's bank with each occurrence present to
    the degree of ``activity`` ``[B, occurrences, 2]``: no lookup and no read
    on the host."""
    owner = model._concept_owner()
    bank = model._sentence_bank
    evidence = owner.cs_read_bank(bank, activity)
    owner._cs_last_a0 = evidence.detach()
    knowing = owner.cs_forward_content(evidence)
    from Understanding import ConceptualField
    return ConceptualField(owner._cs_field_concept_ids,
        F.pad(knowing, (0, 0, 0, 0, 0, 0, 0, sum(owner._field_caps())-len(knowing))),
        owner._cs_field_where, owner._cs_field_when, None, getattr(owner, '_cs_field_wires', None))


def where_match(bank, band_share, activity, readings):
    """Each reading's where match ``[B, readings]``: the share on the where
    wires of the occurrences it holds, each weighted by its presence in the
    look, referred to the most salient reading through the exponential."""
    token = bank.token
    slots = torch.arange(readings, device=token.device)
    held = token[:, None, :] == slots[None, :, None]                      # [B, readings, occurrences]
    presence = activity[..., 0]
    share = torch.where(held, (band_share * presence)[:, None, :], 0.).sum(-1)
    live = held.any(-1)
    best = torch.where(live, share, -torch.inf).amax(-1, keepdim=True)
    return torch.where(live, (share - torch.where(torch.isfinite(best), best, 0.)).exp(), 0.)


def look(model, bank, request, own):
    """One look of ``request`` ``[V, B, 2]`` over ``bank`` with the occurrences'
    own activity ``own`` ``[B, occurrences, 2]``: the returned field and the
    look's activity over the occurrences. The request is carried down by the
    ways back over every occurrence (:func:`Salience.salience`, rule 79); an
    occurrence's activity is its salience bounded by its own activity, the
    look's table is the most salient occurrences (``BasicPercepts.attended``),
    and each reading's values are multiplied once by its where match on that
    salience (:func:`where_match`).
    A look takes no ``.where`` of its own: the ``.where`` is in the concepts."""
    from Salience import salience
    owner = model._concept_owner()
    salient = salience(owner, bank, request)
    # A share is never below zero; at exactly zero its derivative passes, so
    # that a request that starts at nothing is taught through the activity.
    # It bounds the occurrence's own activity, which takes a tie.
    share = salient.events
    activity = torch.where(share < own, share, own)
    field = perceive(model, activity)
    # Each reading's where match, on the salience of what it holds (the
    # ways back give no share of the .where apart from the .what): kept,
    # since without it the request is taught nothing (eighth pass, step 4).
    match = where_match(bank, share.sum(-1), activity, field.evidence.shape[2])
    from dataclasses import replace
    return replace(field, evidence=field.evidence * match[None, :, :, None].to(field.evidence)), activity

def extent_of(bank, activity, readings):
    """The readings a look holds, ``[B, readings]``: those with an
    occurrence active in it."""
    token = bank.token
    lit = (activity.amax(-1) > 0) & (token >= 0)
    slots = torch.arange(readings, device=token.device)
    return ((token[:, None, :] == slots[None, :, None]) & lit[:, None, :]).any(-1)


def word_activity(field, concepts, known, extent, own, bank):
    """Each word's activity in a look's returned set, ``[B, W]``: a word
    with a word concept, that concept's value at its reading (for less
    against), where the look holds the reading. A word with no word concept
    has no concept to return it; it is read whole, at its events' own
    activity in the bank, where the look returned no word. ``-inf`` where a
    word was not found."""
    B, W = concepts.shape
    ids = field.concept_ids
    ids = ids[:, None].expand(-1, B) if ids.ndim == 1 else ids
    evidence = field.evidence[:, :, :W]                                   # [S, B, W, 2]
    value = (evidence[..., 0] - evidence[..., 1])                         # [S, B, W]
    match = (ids[:, :, None] == concepts[None]) & (concepts[None] >= 0)    # [S, B, W]
    concept = torch.where(match, value.to(torch.float32), -torch.inf).amax(0)
    token = bank.token
    slots = torch.arange(W, device=token.device)
    presence = torch.where(token[:, None, :] == slots[None, :, None],
                           own[..., 0][:, None, :], 0.).amax(-1)
    width = min(W, extent.shape[1])
    inside = F.pad(extent[:, :width], (0, W - width))
    word = torch.where(known & inside & (concept > 0), concept, -torch.inf)
    unknown = torch.where(~known & (presence > 0), presence.to(torch.float32), -torch.inf)
    found = torch.isfinite(word).any(-1, keepdim=True)
    return torch.where(found, word, unknown)


def lower(own, bank, index, rows):
    """The done mark: a word interpreted is lowered in the bank's own
    activity over its extent (the occurrences of its reading)."""
    gone = (bank.token == index[:, None]) & rows[:, None] & (index[:, None] >= 0)
    return torch.where(gone[..., None], 0., own)


def attend(model, units):
    """``attend``'s request ``[V, B, 2]``: the read of the conceptual field
    (``model.request_read``, rule 80) from its ``units``."""
    return model.request_read.request(units)


def word_request(model, concepts, index):
    """The request for each stream's word ``index`` ``[B]`` by its word
    concept, for one: a reading supplied as a sequence of requests. A word
    with no word concept has no concept to ask for: its request is empty,
    the look returns no known word, and the word is read whole at its own
    activity (:func:`word_activity`)."""
    owner = model._concept_owner()
    V = int(owner.nVectors)
    request = owner.similarity_codebook.W.new_zeros(V, len(index), 2)
    for b, word in enumerate(index.tolist()):
        cid = int(concepts[b, word]) if word >= 0 else -1
        row = owner._concept_row_of(cid) if cid >= 0 else None
        if row is not None:
            request[row, b, 0] = 1.
    return request


class WordForecast:
    """Shared first-arrival forecast and a ledger at each native word place."""
    def __init__(self, model):
        from ModelAttention import stage_symbol_bank
        from WordExpectation import WordArrivals
        self.model = model
        owner = getattr(model.symbolSpace, 'expectation', None)
        staged = getattr(model, '_attention_forms', None)
        self.reader = None
        if owner is None or staged is None or 'word' not in owner.enabled_levels:
            return
        _, self.identities, keys, self.live = staged
        codes = getattr(model.inputSpace, '_ar_word_object_atoms', None)
        if not torch.is_tensor(codes):
            codes = getattr(model.inputSpace, '_ar_grammar_object_atoms', None)
        self.codes = (keys if not torch.is_tensor(codes) or codes.shape != keys.shape else codes).detach()
        B,W,D = self.codes.shape
        count = max(1, int(self.identities.max())+1)
        matches = (torch.arange(count, device=keys.device)[:,None] == self.identities.reshape(1,-1)) & self.live.reshape(1,-1)
        self.bank = self.codes.reshape(-1,D)[matches.long().argmax(-1)][None].expand(B,-1,-1)
        self.valid = matches.any(-1)[None].expand(B,-1)
        stage_symbol_bank(model, owner)
        self.reader = WordArrivals(owner, self.bank, self.valid, gain=model.word_expectation_gain,
                                  epsilon=model.expectation_epsilon, saturation=model.expectation_saturation)
        self.arrived = torch.zeros_like(self.live)
        self.targets = torch.full_like(self.identities, -1)
        self.expected = None

    def begin_path(self):
        from copy import copy
        # WordArrivals replaces its tensors on every observation. A shallow
        # copy thus shares the immutable opening, never the next path's suffix.
        self.path = copy(self.reader)

    def arrive(self, index, rows, lanes):
        if self.reader is None:
            return
        from WordExpectation import ExpectedWords
        fresh = rows & ~self.arrived[:,index] & self.live[:,index]
        shared = rows & ~fresh & self.live[:,index]
        if bool(shared.any()):
            self.path.observe(self.codes[:,index], shared, lanes)
        if not bool(fresh.any()):
            return
        current = self.path.step(self.codes[:,index],self.identities[:,index],active=fresh,lanes=lanes)
        mask = (torch.arange(self.live.shape[1],device=rows.device)==index)[None] & fresh[:,None]
        values=[]
        for item in fields(ExpectedWords):
            value = getattr(current,item.name)
            prior = (value.new_zeros(value.shape[0],self.live.shape[1],*value.shape[2:])
                     if self.expected is None else getattr(self.expected,item.name))
            condition = mask.reshape(*mask.shape,*([1]*(value.ndim-2)))
            values.append(torch.where(condition,value,prior))
        self.expected = ExpectedWords(*values)
        self.arrived |= mask
        self.targets = torch.where(mask,self.identities,self.targets)
        self.model._word_expectation_input = (self.codes,self.bank,self.valid,self.targets)
        self.model._word_expectation = self.expected
        self.model._word_expectation_mask = self.arrived
        self.model.conceptualSpace._word_column = self.expected.content


class Looks:
    """One path's looks and grammar decisions; words arrive on demand."""
    def __init__(self, model, walk, sid, present, *, prior=None, draw=None, kept=None):
        self.model, self.walk, self.sid, self.present = model, walk, sid, present
        self.prior, self.draw, self.kept = prior, draw, kept
        self.decisions, self.fields, self.stops = [], [], []
        self.supplied = getattr(model, '_attention_supplied_reading', None)

    def source_orders(self):
        """Original word positions in each row's interpretation order."""
        if not self.decisions:
            return tuple([] for _ in range(len(self.present)))
        arrivals = torch.stack([torch.where(turn['interpreted'], turn['index'], -1)
                                for turn in self.decisions], dim=1)
        return tuple([word for word in row if word >= 0] for row in arrivals.tolist())

    def reading(self):
        return self.walk.reading()

    def _supplied(self, state):
        """The next word of a supplied reading, ``[B]`` (-1 where none)."""
        order = self.supplied.positions(self.model._attention_forms[0], state.spans)
        return state.spans.new_tensor([next((w for w in row if not bool(state.accepted[b, w])), -1)
                                       for b, row in enumerate(order)])

    def run(self, arrive):
        model, state = self.model, self.walk
        owner = model._concept_owner()
        opened = model._sentence_open_fields[self.sid]
        bank = model._sentence_bank = model.__dict__.get('_sentence_banks', {}).get(self.sid)
        concepts = model._attention_word_concepts[:, :state.known.shape[1]]
        B, W = state.known.shape
        live = (state.spans[..., 1] > state.spans[..., 0]) & self.present[:, None]
        from LookForecast import LookForecast
        forecasts = LookForecast(owner, opened, state.spans) if opened is not None else None
        own = bank.activity.clone() if bank is not None else None
        field, rows = opened, self.present
        extent = live
        surprise = forecasts.latest if forecasts is not None else None
        from CandidateGrowth import at_look
        at_look(model, field, extent)
        self.fields.append(dict(field=field, extent=extent.clone(), active=rows.clone(), surprise=surprise))
        if bank is None:
            return state.reading()
        # A record that nothing reads (§19.125): each reading's .where
        # wires and code on the .what columns, at its bracket in the bank.
        from Places import readings
        found_where, _, found_code = readings(bank)
        read = model.request_read
        V = int(owner.nVectors)
        from SentenceField import inventory
        known_rows = {}
        def located(field):
            # A field's concepts at their inventory rows, the rows found once
            # for each set of the field's concept ids.
            key = tuple(field.concept_ids.reshape(-1).tolist())
            if key not in known_rows:
                known_rows[key] = inventory(owner, field)
            return at_rows(known_rows[key], field, V, B)
        held = located(opened) if opened is not None else own.new_zeros((B, V, 2))
        memory = torch.zeros_like(held)
        while True:
            turn = state.taken
            left = self.present & (state.limit > turn) & (live & ~state.accepted).any(-1)
            # Whether anything is left to find (the toy's "found": nothing
            # left is what the stop is read from).
            x = read.given(held, memory, left)
            units = read.units(x)
            stop = read.stopping(units)
            self.stops.append(stop.detach())
            # The walk ends where nothing is left, or where the read stops it.
            rows = left & ~(stop.detach() > 0)
            if not bool(rows.any()):
                break
            target = None
            if self.supplied is not None:
                target = self._supplied(state)
                request = word_request(model, concepts, target)
            else:
                request = attend(model, units)
            prediction = None
            look_field, activity = look(model, bank, request, own)
            extent = extent_of(bank, activity, W) & rows[:, None]
            if forecasts is not None:
                prediction = forecasts.forecast(extent, rows)
                # What is expected is written anew at each look, before
                # what the look brings is read.
                from ConceptSpread import write_look
                write_look(forecasts.owner, forecasts.inventory_rows(), prediction)
                surprise = forecasts.observe(look_field, extent, prediction, rows)
                if model.training:
                    looks = model.__dict__.setdefault('_expectation_looks', [])
                    looks.append((forecasts.stage_cost(look_field, extent, rows), rows.clone()))
                    percepts = model.__dict__.setdefault('_expectation_percept_looks', [])
                    perceived = getattr(owner, '_percept_field', None)
                    percepts.append((forecasts.percept_stage_cost(perceived, extent, rows), rows.clone()))
            at_look(model, look_field, extent)
            self.fields.append(dict(field=look_field, extent=extent.clone(), active=rows.clone(),
                                    surprise=surprise))
            words = word_activity(look_field, concepts, state.known, extent, own, bank)
            unread = live & ~state.accepted & rows[:, None]
            words = torch.where(unread, words, -torch.inf)
            replay = masked = None
            exploit = state.exploit
            if exploit is not None and turn < exploit.actions.shape[1]:
                replay = torch.where(turn < state.replay_until, exploit.actions[:, turn], -1)
                masked = torch.where(state.departure == turn, exploit.actions[:, turn], -1)
            if target is not None:
                replay = torch.where(target >= 0, target, -1 if replay is None else replay)
            from Attention import draw_among
            index, drawn, probability, others = draw_among(words, replay=replay, masked=masked)
            index = torch.where(rows, index, -1)
            state.actions.append(index)
            state.alternatives.append(drawn & rows & (index >= 0))
            state.probabilities.append(probability)
            state.counts.append(torch.where(rows, others, 0))
            found = index >= 0
            if not bool(found.any()):
                break
            word = index.clamp_min(0)
            allowed = found & (state.known.gather(1, word[:, None])[:, 0]
                               | torch.isfinite(words.gather(1, word[:, None])[:, 0]))
            state.descended = state.descended | ((torch.arange(W, device=word.device)[None] == word[:, None])
                                                 & (found & ~state.known.gather(1, word[:, None])[:, 0])[:, None])
            options = {}
            decision = len(self.decisions)
            if self.prior is not None and decision < len(self.prior.decisions):
                old = self.prior.decisions[decision]
                if self.kept is not None:
                    replay_grammar = self.present
                else:
                    # A departure in the reading replays the prefix's grammar
                    # decisions; a compose departure keeps the reading's.
                    at = self.draw.get('interpret_round', torch.full_like(index, -1))
                    replay_grammar = (((self.draw['attention_round'] < 0) | (turn < self.draw['attention_round']))
                                      & ((at < 0) | (decision < at)))
                    options['masked_action'] = torch.where(at == decision, old['action'], -1)
                options['replay_action'] = torch.where(replay_grammar, old['action'], -1)
                replay_branch = replay_grammar | (options.get('masked_action', -1) != -1)
                options['replay_bias'] = torch.where(replay_branch, old['biased'].long(), -1)
            choice = state.chooser.interpret_at_look(state.keys[torch.arange(len(word), device=word.device), word],
                                                     allowed, **options)
            choice.update(index=index, slot=index.clamp_min(0))
            # The record of the look (§19.125): the request it was given, and
            # the .where and the code of the word it took (nothing where none);
            # the chance of stopping read before the look.
            taken = index.clamp(0, found_where.shape[1] - 1)
            stream = torch.arange(len(index), device=index.device)
            choice.update(request=request.detach(),
                          where=torch.where(found[:, None], found_where[stream, taken].detach(), 0.),
                          code=torch.where(found[:, None], found_code[stream, taken].detach(), 0.),
                          stop=stop.detach(), scores=words.detach())
            self.decisions.append(choice)
            interpreted = choice['interpreted'] & found
            # The grammar's option to take the basic level's read of the
            # run of words that begins here as one constituent.
            from FragmentRead import take
            replayed = None
            if self.prior is not None and decision < len(self.prior.decisions):
                replayed = self.prior.decisions[decision].get('fragment_taking')
            fragment = take(model, state, bank, extent, index, interpreted, prior=replayed)
            if fragment is not None:
                choice['fragment_taking'] = fragment['taking']
                choice['fragment'] = fragment['run']
            state.record(word, interpreted)
            own = lower(own, bank, index, interpreted)
            if fragment is not None and fragment['idea'] is not None:
                others = fragment['run'] & (torch.arange(W, device=index.device)[None] != index[:, None])
                for value in others.any(0).nonzero().flatten().tolist():
                    with_it = others[:, value]
                    at = torch.full_like(index, value)
                    state.record(at, with_it)
                    own = lower(own, bank, at, with_it)
                model._fragment_leaf = dict(index=index, taking=fragment['taking'],
                                            idea=fragment['idea'], classes=fragment['classes'])
            # What the next look reads: this look's field spread once.
            memory = torch.where(rows[:, None, None], spread_once(owner, located(look_field)), memory)
            handoff_reading(model, state)
            model._sentence_look_fields = dict(getattr(model, '_sentence_look_fields', {}))
            model._sentence_look_fields[self.sid] = tuple(self.fields)
            # Batch rows can interpret different words at the same look.
            try:
                for value in index[interpreted].unique().tolist():
                    arrive(torch.tensor(value, device=index.device), interpreted & (index == value))
            finally:
                model.__dict__.pop('_fragment_leaf', None)
        return state.reading()


def handoff_reading(model, state):
    from ModelAttention import handoff
    handoff(model, state.reading())
