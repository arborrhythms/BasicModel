"""A sentence's looks: each a request carried down onto the sentence's
percept bank, read up again, and what it returns handed to the grammar.

A look (§21, rules 34, 37 and 39): an occurrence's activity is its share of
the request capped at its own activity; a reading's where match is
``exp(s - max s)`` of the share on the where wires of what it holds, applied
once to the reading's values; the way up again on a table of the bank chosen
by that activity gives the returned set. ``interpret`` takes the most active word, and a word
interpreted is lowered in the bank's own activity over its extent.
"""
from dataclasses import fields
import torch
from torch.nn import functional as F


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
        owner._cs_field_where, owner._cs_field_when, None)


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


def look(model, bank, request, own, perception):
    """One look of ``request`` ``[V, B, 2]`` over ``bank`` with the occurrences'
    own activity ``own`` ``[B, occurrences, 2]``: the returned field, the look's
    activity over the occurrences, and each reading's where match."""
    from Salience import salience
    owner = model._concept_owner()
    salient = salience(owner, bank, request, perception=perception)
    # A share is never below zero; at exactly zero its derivative passes, so
    # that a request that starts at nothing is taught through the activity.
    # It bounds the occurrence's own activity, which takes a tie.
    share = torch.where(salient.events >= 0, salient.events, 0.)
    activity = torch.where(share < own, share, own)
    field = perceive(model, activity)
    readings = field.evidence.shape[2]
    match = where_match(bank, salient.band, activity, readings)
    from dataclasses import replace
    field = replace(field, evidence=field.evidence * match[None, :, :, None].to(field.evidence))
    return field, activity, match


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


def requested(model, query):
    """``attend``'s request ``[V, B, 2]`` from the context's ``query``
    ``[B, D]``: the learned start (``model.request_start``), and each concept
    as its code reads the query."""
    owner = model._concept_owner()
    # The codes of the rows that hold a concept: the read of the query keeps
    # them for its backward at every look, and a row with no concept is
    # asked by the start alone.
    from Spaces import _concept_alloc_of
    held = sorted(set(_concept_alloc_of(owner).layer(0)._tensor_rows.values()))
    V = int(owner.nVectors)
    rows = torch.tensor([row for row in held if 0 <= row < V], dtype=torch.long, device=query.device)
    codes = owner.similarity_codebook.W.detach().index_select(0, rows.to(owner.similarity_codebook.W.device))
    return model._stm_reducer().attend(query, codes, model.request_start.value, rows)


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
        self.decisions, self.fields = [], []
        self.teacher = None
        lesson = getattr(model, '_attention_reading_lesson', None)
        if lesson is not None and model.training and model._sentence_training:
            from AttentionLesson import ReadingTargets
            self.teacher = ReadingTargets(lesson, model._attention_forms[0], walk)
        self.supplied = getattr(model, '_attention_supplied_reading', None)

    def source_orders(self):
        """Original word positions in each row's interpretation order."""
        if not self.decisions:
            return tuple([] for _ in range(len(self.present)))
        arrivals = torch.stack([torch.where(turn['interpreted'], turn['index'], -1)
                                for turn in self.decisions], dim=1)
        return tuple([word for word in row if word >= 0] for row in arrivals.tolist())

    def lesson_cost(self):
        return None if self.teacher is None else self.teacher.cost()

    def reading(self):
        return self.walk.reading()

    def _supplied(self, state):
        """The next word of a supplied reading, ``[B]`` (-1 where none)."""
        order = self.supplied.positions(self.model._attention_forms[0], state.spans)
        return state.spans.new_tensor([next((w for w in row if not bool(state.accepted[b, w])), -1)
                                       for b, row in enumerate(order)])

    def run(self, arrive, *, context=None):
        model, state = self.model, self.walk
        owner = model._concept_owner()
        opened = model._sentence_open_fields[self.sid]
        bank = model._sentence_bank = model.__dict__.get('_sentence_banks', {}).get(self.sid)
        concepts = model._attention_word_concepts[:, :state.known.shape[1]]
        W = state.known.shape[1]
        live = (state.spans[..., 1] > state.spans[..., 0]) & self.present[:, None]
        from LookForecast import LookForecast
        forecasts = LookForecast(owner, opened, state.spans) if opened is not None else None
        perception = owner.basic_percepts.read_bank(bank) if bank is not None else None
        own = bank.activity.clone() if bank is not None else None
        field, rows = opened, self.present
        extent = live
        surprise = forecasts.latest if forecasts is not None else None
        from CandidateGrowth import at_look
        at_look(model, field, extent)
        self.fields.append(dict(field=field, extent=extent.clone(), active=rows.clone(), surprise=surprise))
        while bank is not None:
            turn = state.taken
            rows = self.present & (state.limit > turn) & (live & ~state.accepted).any(-1)
            if not bool(rows.any()):
                break
            target = None
            if self.teacher is not None:
                target = self.teacher.next(state)
            elif self.supplied is not None:
                target = self._supplied(state)
            if self.supplied is not None and self.teacher is None:
                request = word_request(model, concepts, target)
            else:
                query = (context(surprise[:, None])[:, 0] if context is not None and surprise is not None
                         else owner.similarity_codebook.getW().new_zeros(len(rows), int(model._stm_reducer().d_model)))
                request = requested(model, query)
            prediction = None
            look_field, activity, match = look(model, bank, request, own, perception)
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
                                    surprise=surprise, match=match.detach()))
            words = word_activity(look_field, concepts, state.known, extent, own, bank)
            unread = live & ~state.accepted & rows[:, None]
            words = torch.where(unread, words, -torch.inf)
            if self.teacher is not None:
                self.teacher.book_look(words, target, unread)
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
            if self.teacher is not None:
                options['replay_action'] = allowed.long()
                options.pop('masked_action', None)
            choice = state.chooser.interpret_at_look(state.keys[torch.arange(len(word), device=word.device), word],
                                                     allowed, **options)
            if self.teacher is not None:
                logp = choice['logits'].log_softmax(-1).gather(1, choice['action'][:, None])[:, 0]
                self.teacher.book(logp, choice['eligible'] & rows)
                choice['eligible'] = torch.zeros_like(allowed)
            choice.update(index=index, slot=index.clamp_min(0))
            self.decisions.append(choice)
            interpreted = choice['interpreted'] & found
            fragment = None
            if self.teacher is None:
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
