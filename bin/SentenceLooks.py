"""A sentence's looks: each a request carried down onto the sentence's
percept bank, read up again, and what it returns handed to the grammar.

A look (§21, rules 34, 37, 39, 79 and 80): the request is ``attend``'s read
of the conceptual field (:mod:`RequestRead`); an occurrence's activity is
its salience, the request carried down by the ways back, capped at its own
activity; a reading's where match is ``exp(s - max s)`` of the salience of
what it holds, applied once to the reading's values; the way up again on a
table of the bank's most salient occurrences gives the returned set. The
request assigns selection probabilities to the eligible words, and a word interpreted is
lowered in the bank's own activity over its extent.
"""
from dataclasses import fields
import torch
from torch.nn import functional as F

from BasicPercepts import reading_scope
from RequestRead import at_rows, spread_once


def perceive(model, activity, bank, *, fixation=None):
    """A look's read of the sentence's bank with each occurrence present to
    the degree of ``activity`` ``[B, occurrences, 2]``: no lookup and no read
    on the host. The unit delay on basic percepts does not keep a look's
    read: they step from opening to opening (plan §9.5)."""
    owner = model._concept_owner()
    evidence = (owner.cs_read_bank(bank, activity, keep=False) if fixation is None
                else owner.cs_read_bank(bank, activity, fixation=fixation, keep=False))
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
    held = reading_scope(bank, readings)
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
    field = perceive(model, activity, bank)
    # Each reading's where match, on the salience of what it holds (the
    # ways back give no share of the .where apart from the .what): kept,
    # since without it the request is taught nothing (eighth pass, step 4).
    match = where_match(bank, share.sum(-1), activity, field.evidence.shape[2])
    return field.scaled_at_locations(match.to(field.evidence)), activity

def extent_of(bank, activity, readings):
    """The readings a look holds, ``[B, readings]``: those with an
    occurrence active in it."""
    lit = activity.amax(-1) > 0
    return (reading_scope(bank, readings) & lit[:, None]).any(-1)


def word_activity(field, concepts, known, extent, own, bank, general=False):
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
    presence = torch.where(reading_scope(bank, W), own[..., 0][:, None], 0.).amax(-1)
    width = min(W, extent.shape[1])
    inside = F.pad(extent[:, :width], (0, W - width))
    word = torch.where(known & inside & (concept > 0), concept, -torch.inf)
    unknown = torch.where(~known & (presence > 0), presence.to(torch.float32), -torch.inf)
    if general:
        return torch.where(known, word, unknown)
    found = torch.isfinite(word).any(-1, keepdim=True)
    return torch.where(found, word, unknown)


def narrow(model, bank, request, own, concepts, known, eligible):
    """Read the requested field and score the still eligible occurrences on that field."""
    rows = model.request_read.rows
    general = len(request) > rows
    prior = own.new_zeros(concepts.shape)
    if general:
        from RequestRead import spatial_match
        prior = spatial_match(bank, request[rows:, :, 0].T, readings=concepts.shape[1])
    # A located hypothesis has the query's spatial log weight and the
    # returned whole's content evidence. Apply the spatial weight once.
    # Feeding it through the percept detector as well could hide a real
    # occurrence before its evidence is compared to the expectation.
    field, activity = look(model, bank, request[:rows], own)
    extent = extent_of(bank, activity, concepts.shape[1]) & eligible.any(-1, keepdim=True)
    evidence = word_activity(field, concepts, known, extent, own, bank, general)
    scores = (prior + evidence).masked_fill(~eligible, -torch.inf)
    if general:
        # Keep the field under the same spatial constraint for the following
        # look. Eligibility and the production of evidence preceded this cap.
        supported = prior.masked_fill(~scores.isfinite(), -torch.inf)
        best = supported.amax(1, keepdim=True)
        strength = (supported - torch.where(best.isfinite(), best, 0.)).exp()
        strength = F.pad(strength, (0, field.evidence.shape[2] - strength.shape[1]))
        field = field.scaled_at_locations(strength)
    return field, extent, scores


def lower(own, bank, index, rows):
    """The done mark: a word interpreted is lowered in the bank's own
    activity over its extent (the occurrences of its reading)."""
    gone = (bank.token == index[:, None]) & rows[:, None] & (index[:, None] >= 0)
    return torch.where(gone[..., None], 0., own)


def attend(model, units):
    """Predict one located concept from the open field and STM context."""
    return model.request_read(units)


def request_occurrences(model, bank, request, own, eligible):
    """The neural request's scores at every observed occurrence of the sentence."""
    from RequestRead import spatial_match
    from Salience import salience
    rows = model.request_read.rows
    share = salience(model._concept_owner(), bank, request[:rows]).events
    activity = torch.where(share < own, share, own)
    score = activity.sum(-1)
    responsive = score > 0
    if request.shape[0] > rows:
        prior = spatial_match(bank, request[rows:, :, 0].T)
        responsive = responsive | (prior > 0)
        score = score + prior
    live = eligible & (own[..., 0] > 0)
    # A first encounter can have no learned response anywhere. The observed
    # percepts still provide the same sensory fallback as an unnamed whole.
    found = live & responsive
    score = torch.where(found.any(-1, keepdim=True), score.masked_fill(~found, -torch.inf), own[..., 0])
    return score.masked_fill(~live, -torch.inf), activity


def word_request(model, concepts, index):
    """The request for each stream's word ``index`` ``[B]`` by its word
    concept, for one: a reading supplied as a sequence of requests. A word
    with no word concept has no concept to ask for: its request is empty,
    the look returns no known word, and the word is read whole at its own
    activity (:func:`word_activity`)."""
    owner = model._concept_owner()
    V = int(owner.nVectors)
    request = owner.similarity_codebook.W.new_zeros(V, len(index), 2)
    if not V or not concepts.shape[1]:
        return request
    source = index.clamp_min(0)[:, None]
    staged = getattr(getattr(model, 'perceptualSpace', None), '_forward_input', None) or {}
    seats = staged.get('pool_concept_rows')
    words = concepts.gather(1, source).flatten()
    if seats is None:
        rows = owner._concept_allocator.layer(0).concept_rows(torch.where(index >= 0, words, -1))
    else:
        rows = torch.where((index >= 0) & (words >= 0), seats.gather(1, source).flatten(), -1)
    stream = torch.arange(len(index), device=request.device)
    request[rows.clamp_min(0), stream, 0] = (rows >= 0).to(request)
    return request


class WordForecast:
    """One immutable opening and a separate causal word history for each path."""
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
        self.bank = self.codes
        self.valid = self.live & (self.identities == torch.arange(keys.shape[1], device=keys.device))
        stage_symbol_bank(model, owner)
        self.reader = WordArrivals(owner, self.bank, self.valid, gain=model.word_expectation_gain,
                                  epsilon=model.expectation_epsilon, saturation=model.expectation_saturation)
    def begin_path(self, walk=None):
        from copy import copy
        path = copy(self)
        path.walk = walk
        path.predicted_look = None
        if self.reader is not None:
            # WordArrivals replaces its context tensors on observation.
            path.reader = copy(self.reader)
            path.arrived = torch.zeros_like(self.live)
            path.targets = torch.full_like(self.identities, -1)
            path.expected = None
            path.publish()
        return path

    def publish(self):
        if self.reader is None:
            return
        self.model._word_expectation_input = (self.codes, self.bank, self.valid, self.targets)
        self.model._word_expectation = self.expected
        self.model._word_expectation_mask = self.arrived
        self.model.conceptualSpace._word_column = None if self.expected is None else self.expected.content

    @staticmethod
    def keep(paths, winners):
        from copy import copy
        from WordExpectation import ExpectedWords
        first = paths[0]
        if first.reader is None:
            return
        if all(path is first for path in paths):
            first.publish()
            return
        retained = copy(first)
        streams = torch.arange(len(winners), device=winners.device)
        def choose(values):
            return torch.stack(values, dim=1)[streams, winners]
        retained.targets = choose([path.targets for path in paths])
        retained.arrived = choose([path.arrived for path in paths])
        template = next((path.expected for path in paths if path.expected is not None), None)
        if template is not None:
            retained.expected = ExpectedWords(*(choose([
                getattr(path.expected, name) if path.expected is not None
                else torch.zeros_like(getattr(template, name)) for path in paths])
                for name in ('probabilities', 'prediction', 'content', 'gain', 'excess',
                             'omission', 'loss', 'top1', 'reciprocal_rank', 'activation')))
        retained.publish()

    def arrive(self, index, rows, lanes):
        if self.reader is None:
            return
        from WordExpectation import ExpectedWords
        per_batch = torch.is_tensor(index) and index.ndim > 0
        W = self.live.shape[1]
        if per_batch:
            ar = torch.arange(len(rows), device=rows.device)
            safe = index.clamp(0, W - 1)
            fresh = rows & ~self.arrived[ar, safe] & self.live[ar, safe]
            codes = self.codes[ar, safe]
            identities = self.identities[ar, safe]
            live_here = rows & self.live[ar, safe]
            mask = (torch.arange(W, device=rows.device)[None] == index[:, None]) & fresh[:, None]
        else:
            fresh = rows & ~self.arrived[:,index] & self.live[:,index]
            codes = self.codes[:, index]
            identities = self.identities[:, index]
            live_here = rows & self.live[:, index]
            mask = (torch.arange(W,device=rows.device)==index)[None] & fresh[:,None]
        # Score from this path's prefix. A repeated occurrence still enters
        # its observed context, but only its first forecast occupies the ledger.
        # The interpreter dispatches disjoint sets of streams by source
        # column within one look. Each stream has only one arrival at this
        # look. All columns can read the same batch of pre-arrival logits.
        turn = None if self.walk is None else self.walk.taken
        regions = getattr(self.model, '_compiled_sentence_regions', None)
        step_fn = (regions['word_arrivals_step']
                   if regions and not torch.compiler.is_compiling() else type(self.reader).step)
        current = step_fn(self.reader, codes, identities,
                          active=live_here, lanes=lanes,
                          forecast=turn is None or turn != self.predicted_look)
        self.predicted_look = turn
        values=[]
        for name in ExpectedWords.__dataclass_fields__:
            value = getattr(current, name)
            prior = (value.new_zeros(value.shape[0],self.live.shape[1],*value.shape[2:])
                     if self.expected is None else getattr(self.expected, name))
            condition = mask.reshape(*mask.shape,*([1]*(value.ndim-2)))
            values.append(torch.where(condition,value,prior))
        self.expected = ExpectedWords(*values)
        self.arrived = self.arrived | mask
        self.targets = torch.where(mask,self.identities,self.targets)
        self.publish()


class Looks:
    """One path's looks and grammar decisions; words arrive on demand."""
    def __init__(self, model, walk, sid, present, *, prior=None, draw=None, kept=None):
        self.model, self.walk, self.sid, self.present = model, walk, sid, present
        self.prior, self.draw, self.kept = prior, draw, kept
        self.decisions, self.fields, self.stops = [], [], []
        self._turn_slots = []
        self.look_credit = None
        self.open_forecast = None
        # This path's kept field (the unit delay): look zero, and at the
        # last look what was one step back, observed and primed.
        self.opening_primed = self.opening_before = self.final = None
        self.supplied = getattr(model, '_attention_supplied_reading', None)

    def source_orders(self):
        """Original word positions in each row's interpretation order."""
        if not self.decisions:
            return tuple([] for _ in range(len(self.present)))
        if len(self._turn_slots) == len(self.decisions):
            return tuple([word for word in row if word >= 0] for row in zip(*self._turn_slots))
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

    def _look_interpret_and_spread(self, model, owner, bank, state, observation,
                                   look_field, extent, words, rows, index, found,
                                   anticipated, request, stop, found_where, found_code,
                                   own, memory, stream, word_slots, decision, turn, primed=None):
        """Interpret the look's word and update what the next request reads
        of this look: with no unit delay, ``memory``, the field at the
        reading taken spread once; with it, ``primed``, the path's kept
        field ``(observed, trace)``: the field observed at the reading taken
        (the delay) and what the look's cycle leaves, masked on the way down
        by what ``request`` asked (:func:`RequestRead.prime`); ``memory`` is
        the trace."""
        if self.look_credit is not None:
            self.look_credit.observe(
                anticipated,
                torch.where(found[:, None], observation.last_where, -1))
        word = index.clamp_min(0)
        allowed = found & (state.known.gather(1, word[:, None])[:, 0]
                           | torch.isfinite(words.gather(1, word[:, None])[:, 0]))
        state.descended = state.descended | ((word_slots == word[:, None])
                                             & (found & ~state.known.gather(1, word[:, None])[:, 0])[:, None])
        options = {}
        if self.prior is not None and decision < len(self.prior.decisions):
            old = self.prior.decisions[decision]
            if self.kept is not None:
                replay_grammar = self.present
            else:
                at = self.draw.get('interpret_round', torch.full_like(index, -1))
                replay_grammar = (((self.draw['attention_round'] < 0) | (turn < self.draw['attention_round']))
                                  & ((at < 0) | (decision < at)))
                options['masked_action'] = torch.where(at == decision, old['action'], -1)
            options['replay_action'] = torch.where(replay_grammar, old['action'], -1)
            replay_branch = replay_grammar | (options.get('masked_action', -1) != -1)
            options['replay_bias'] = torch.where(replay_branch, old['biased'].long(), -1)
        choice = state.chooser.interpret_at_look(state.keys[stream, word], allowed, **options)
        choice.update(index=index, slot=index.clamp_min(0))
        taken = index.clamp(0, found_where.shape[1] - 1)
        choice.update(request=request.detach(),
                      where=torch.where(found[:, None], found_where[stream, taken].detach(), 0.),
                      code=torch.where(found[:, None], found_code[stream, taken].detach(), 0.),
                      stop=stop.detach(), scores=words.detach())
        interpreted = choice['interpreted'] & found
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
        fragment_leaf = None
        if fragment is not None and fragment['idea'] is not None:
            others = fragment['run'] & (word_slots != index[:, None])
            state.accepted = state.accepted | others
            state.values = torch.where(others[..., None], state.keys, state.values)
            extra = F.pad(others, (0, 1))
            token = bank.token
            W = state.known.shape[1]
            taken_occurrences = extra.gather(1, torch.where((token >= 0) & (token < W), token, W))
            own = torch.where(taken_occurrences[..., None], 0., own)
            fragment_leaf = dict(index=index, taking=fragment['taking'],
                                 idea=fragment['idea'], classes=fragment['classes'])
        V = int(owner.nVectors)
        from SentenceField import inventory
        observed = at_rows(inventory(owner, look_field), look_field, V, len(rows), index)
        if primed is None:
            memory = torch.where(rows[:, None, None], spread_once(owner, observed), memory)
        else:
            from RequestRead import prime
            from Salience import concepts_asked
            kept, trace = primed
            # What attend predicted at this look, carried down: the mask on
            # the way down of the look's cycle.
            asked = concepts_asked(owner, request[:model.request_read.rows].detach()).transpose(0, 1)
            trace = prime(model, owner, observed, trace, rows, asked=asked)
            primed = (torch.where(rows[:, None, None], observed, kept), trace)
            memory = trace
        return choice, interpreted, own, memory, primed, fragment_leaf

    def run(self, arrive, stack=None):
        """The path's looks. ``arrive(word, rows)`` hands a word taken to the
        grammar, which shifts it on the STM stack and reduces; ``stack()``
        is this path's STM as it stands, which each look's request reads:
        at the first look as the sentence begins, after it as the last
        look's reduces left it."""
        model, state = self.model, self.walk
        owner = model._concept_owner()
        opened = model._sentence_open_fields[self.sid]
        bank = model._sentence_bank = model.__dict__.get('_sentence_banks', {}).get(self.sid)
        concepts = model._attention_word_concepts[:, :state.known.shape[1]]
        B, W = state.known.shape
        live = (state.spans[..., 1] > state.spans[..., 0]) & self.present[:, None]
        from LookForecast import LookForecast
        # All departures start at this sentence's same wide-open observation.
        # observe() replaces a walk's history tensors as later looks arrive.
        from copy import copy
        if opened is not None:
            self.open_forecast = (self.prior.open_forecast if self.prior is not None
                                  else LookForecast(owner, opened, state.spans))
        forecasts = copy(self.open_forecast) if self.open_forecast is not None else None
        own = bank.activity.clone() if bank is not None else None
        observation = model.conceptualSpace.stm.model_state
        observation.begin_looks(state.spans, self.present)
        field, rows = opened, self.present
        extent = live
        surprise = forecasts.latest if forecasts is not None else None
        from CandidateGrowth import at_look, at_opening
        nominees = getattr(getattr(owner, 'mtm', None), '_field_nominees', None)
        self.opening_growth = at_opening(model, field, extent, rows,
            prior=None if self.prior is None else self.prior.opening_growth, binding=nominees)
        self.fields.append(dict(field=field, extent=extent.clone(), active=rows.clone(), surprise=surprise))
        if bank is None:
            return state.reading()
        # A record that nothing reads (§19.125): each reading's .where
        # wires and code on the .what columns, at its bracket in the bank.
        from Places import readings
        if self.prior is None:
            found_where, _, found_code = readings(bank)
            self.observed_readings = found_where.detach(), found_code.detach()
        else:
            self.observed_readings = self.prior.observed_readings
        found_where, found_code = self.observed_readings
        read = model.request_read
        V = int(owner.nVectors)
        from SentenceField import inventory
        def located(field, index=None):
            return at_rows(inventory(owner, field), field, V, B, index)
        held = located(opened) if opened is not None else own.new_zeros((B, V, 2))
        memory = torch.zeros_like(held)
        primed = self._opening_primed(model, owner, B, held) if owner.percept_delay else None
        if primed is not None:
            # The opening is look zero: the first look reads its trace.
            memory = primed[1]
        if model.training and torch.is_grad_enabled() and self.supplied is None:
            from LookForecast import OccurrenceCredit
            self.look_credit = (self.prior.look_credit.fork() if self.prior is not None and self.prior.look_credit is not None
                                else OccurrenceCredit(owner, bank, choose_extent=False))
        stream = torch.arange(B, device=own.device)
        word_slots = torch.arange(W, device=own.device)[None]
        compiling = torch.compiler.is_compiling()
        regions = None if compiling else getattr(model, '_compiled_sentence_regions', None)
        narrow_fn = regions['narrow'] if regions else narrow
        look_interpret_fn = regions['look_interpret_and_spread'] if (regions and 'look_interpret_and_spread' in regions) else Looks._look_interpret_and_spread
        hardwire_reading = bool(getattr(model, 'hardwire_reading', False))
        look_iters = (
            (range(max(1, int(model.attention_budget))) if compiling else iter(int, 1))
            if hardwire_reading else range(1))
        for _ in look_iters:
            turn = state.taken
            left = self.present & (state.limit > turn) & (live & ~state.accepted).any(-1)
            # Whether anything is left to find (the toy's "found": nothing
            # left is what the stop is read from).
            x = read.given(held, memory, left)
            units = read.units(x, observation.request_state(model.where_registry),
                               None if stack is None else stack())
            stop = read.stopping(units)
            self.stops.append(stop.detach())
            # The walk ends where nothing is left, or where the read stops it.
            rows = left & ~(stop.detach() > 0)
            if not compiling and not bool(rows.any()):
                break
            if primed is not None:
                # The concepts' step is the look: what is one step back is
                # the field observed at the look before, this path's own.
                with torch.no_grad():
                    owner._concept_delay_previous.copy_(primed[0].transpose(0, 1))
                    owner._concept_delay_current.zero_()
            if regions:
                from CandidateGrowth import visible_store
                from ConceptSpread import definition_depth
                _store = visible_store(owner._concept_allocator.layer(0))
                _V = int(owner.nVectors)
                _store.spread_indices(_V, own.device)
                definition_depth(_store, _V, _store.inventory_indices(_V, own.device))
            target = None
            if self.supplied is not None:
                target = self._supplied(state)
                request = word_request(model, concepts, target)
            else:
                request = attend(model, units)
            anticipated = self.look_credit.forecast(request) if self.look_credit is not None else None
            unread = live & ~state.accepted & rows[:, None]
            look_field, extent, words = narrow_fn(model, bank, request, own, concepts, state.known, unread)
            prediction = None
            if forecasts is not None:
                prediction = forecasts.forecast(extent, rows)
                from ConceptSpread import write_look
                write_look(forecasts.owner, forecasts.inventory_rows(), prediction)
                surprise = forecasts.observe(
                    look_field, extent, prediction, rows,
                    _core_fn=regions['look_forecast_observe'] if regions else None)
                if model.training:
                    perceived = getattr(owner, '_percept_field', None)
                    concept_cost, percept_cost = forecasts.stage_costs(look_field, perceived, extent, rows)
                    model.__dict__.setdefault('_expectation_looks', []).append((concept_cost, rows.clone()))
                    model.__dict__.setdefault('_expectation_percept_looks', []).append((percept_cost, rows.clone()))
            nominees = getattr(getattr(owner, 'mtm', None), '_field_nominees', None)
            # MTM reads only kind rows. These are views of this look's
            # forecast and stay valid while other paths run.
            kind_forecast = None if prediction is None else (
                forecasts.higher_rows, prediction[:, forecasts.order_zero_width:, 0])
            at_look(model, look_field, extent, rows, forecast=kind_forecast,
                    binding=nominees)
            self.fields.append(dict(field=look_field, extent=extent.clone(), active=rows.clone(),
                                    surprise=surprise))
            replay = masked = None
            exploit = state.exploit
            if exploit is not None and turn < exploit.actions.shape[1]:
                replay = torch.where(turn < state.replay_until, exploit.actions[:, turn], -1)
                masked = torch.where(state.departure == turn, exploit.actions[:, turn], -1)
            if target is not None:
                replay = torch.where(target >= 0, target, -1 if replay is None else replay)
            from Attention import occurrence_probabilities, select_occurrence
            probabilities = occurrence_probabilities(words)
            index, proposal = select_occurrence(probabilities, sample=model.training, replay=replay,
                                                masked=masked, return_proposal=True)
            others = (probabilities.detach().gt(0).sum(-1) - 1).clamp_min(0)
            probability = probabilities.gather(1, index.clamp_min(0)[:, None])[:, 0]
            index = torch.where(rows, index, -1)
            observation.observe(bank.where, index)
            state.record_look(index, (others > 0) & rows & (index >= 0) & (self.supplied is None),
                              probability, torch.where(rows, proposal, 0.))
            found = index >= 0
            if not compiling and not bool(found.any()):
                break
            decision = len(self.decisions)
            before = None if primed is None else primed[0]
            choice, interpreted, own, memory, primed, fragment_leaf = look_interpret_fn(
                self, model, owner, bank, state, observation,
                look_field, extent, words, rows, index, found,
                anticipated, request, stop, found_where, found_code,
                own, memory, stream, word_slots, decision, turn, primed)
            if primed is not None:
                self.final = (before, primed[0], primed[1])
                with torch.no_grad():
                    owner._concept_delay_current.copy_(primed[0].transpose(0, 1))
            self.decisions.append(choice)
            if fragment_leaf is not None:
                model._fragment_leaf = fragment_leaf
            handoff_reading(model, state)
            model._sentence_look_fields = dict(getattr(model, '_sentence_look_fields', {}))
            model._sentence_look_fields[self.sid] = tuple(self.fields)
            # Batch rows can interpret different words at the same look.
            try:
                # Each stream offers at most one source slot per look. Keep
                # the batch width fixed at the serial grammar handoff.
                if compiling and getattr(model, '_canonical_sentence_bricks', False):
                    arrive(index, interpreted)
                elif compiling:
                    for word in range(W):
                        arrive(word, interpreted & (index == word))
                else:
                    slots = torch.where(interpreted, index, -1).tolist()
                    self._turn_slots.append(slots)
                    for word in sorted(set(slots) - {-1}):
                        arrive(word, interpreted & (index == word))
            finally:
                model.__dict__.pop('_fragment_leaf', None)
        if primed is not None:
            Looks._commit(owner, self.final)
        return state.reading()

    def _opening_primed(self, model, owner, B, held):
        """This sentence's look zero, ``(observed, trace)``, each ``[B, V,
        2]``: the field kept at the opening, and what the opening's cycle
        leaves from the trace the last sentence left, attention fully open
        (:func:`RequestRead.prime`). Every path of the sentence starts from it, so
        a path's looks read nothing another path wrote; it is made once, by
        the first path."""
        if self.prior is not None and getattr(self.prior, 'opening_primed', None) is not None:
            self.opening_primed, self.opening_before = self.prior.opening_primed, self.prior.opening_before
        else:
            from RequestRead import prime
            owner._make_delay('concept', held.shape[1], B, held.device)
            observed = owner._concept_delay_opening.transpose(0, 1).clamp(0., 1.).clone()
            carried = owner._concept_trace.transpose(0, 1).clone()
            self.opening_primed = (observed, prime(model, owner, observed, carried))
            # What the opening had one step back: the opening before it.
            self.opening_before = owner._concept_delay_previous.transpose(0, 1).clone()
        self.final = (self.opening_before, *self.opening_primed)
        return self.opening_primed

    @staticmethod
    @torch.no_grad()
    def _commit(owner, final):
        """Write a path's kept field into the unit delay: what is one step
        back, what the last look observed, and the trace."""
        before, observed, trace = final
        owner._concept_delay_previous.copy_(before.transpose(0, 1))
        owner._concept_delay_current.copy_(observed.transpose(0, 1))
        owner._concept_trace.copy_(trace.transpose(0, 1))

    @staticmethod
    def keep(paths, winners):
        """At a sentence's end the kept field of each stream's winning path
        (``winners`` ``[B]``, an index into ``paths``) is what the unit delay
        holds; a discarded path leaves nothing in it."""
        held = [getattr(path, 'final', None) for path in paths]
        first = next((value for value in held if value is not None), None)
        if first is None:
            return
        owner = next(path for path in paths if getattr(path, 'final', None) is not None).model._concept_owner()
        held = [first if value is None else value for value in held]
        streams = torch.arange(len(winners), device=winners.device)
        Looks._commit(owner, tuple(torch.stack([value[part] for value in held], 1)[streams, winners]
                                   for part in range(3)))


def handoff_reading(model, state):
    from ModelAttention import handoff
    handoff(model, state.reading())
