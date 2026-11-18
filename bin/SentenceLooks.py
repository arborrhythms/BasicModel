"""Perceive each selected extent before grammar considers its word."""
from dataclasses import fields
import torch
from torch.nn import functional as F


def perceive(model, interval, active):
    """The open field's native read, now at one selected bracket per stream."""
    owner, ps, ws = model._concept_owner(), model.perceptualSpace, model.wholeSpace
    raw, native = model._staged_concepts_in, ps._forward_input
    bracket = torch.where(active[:, None, None], interval[:, None], 0)
    owner._passback_scope_where = interval.to(raw) / max(1, raw.shape[-1])
    owner._passback_scope_space = interval.new_zeros(len(interval))
    whole_spans = getattr(ws, '_staged_word_property_spans', None)
    whole_spans = (ws.concept_evidence_layout(raw, ws.inputShape[0])[0] if whole_spans is None
                   else whole_spans.reshape(raw.shape[0], -1, 2))
    evidence = owner.cs_read_memberships((native.get('native_indices', native['indices']),
        native.get('native_part_spans', native['part_spans']),
        ws.subspace.what.primitive_properties, raw, whole_spans), bracket)
    owner._cs_last_a0 = evidence.detach()
    _, knowing = owner.cs_forward_content(evidence, None)
    from Understanding import ConceptualField
    field = ConceptualField(owner._cs_field_concept_ids,
        F.pad(knowing, (0, 0, 0, 0, 0, 0, 0, sum(owner._field_caps())-len(knowing))),
        owner._cs_field_where, owner._cs_field_when, None)
    return field


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

    def arrive(self, index, rows, lanes):
        if self.reader is None:
            return
        from WordExpectation import ExpectedWords
        fresh = rows & ~self.arrived[:,index] & self.live[:,index]
        if not bool(fresh.any()):
            return
        current = self.reader.step(self.codes[:,index],self.identities[:,index],active=fresh,lanes=lanes)
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
    """One path's ordered looks and grammar decisions; words arrive on demand."""
    def __init__(self, model, walk, sid, present, *, prior=None, draw=None, kept=None):
        self.model, self.walk, self.sid, self.present = model, walk, sid, present
        self.prior, self.draw, self.kept = prior, draw, kept
        self.decisions, self.fields = [], []

    def reading(self):
        from Attention import close_narrowing
        from ModelAttention import _word_carrier
        return _word_carrier(close_narrowing(self.walk.state),self.walk.width)

    def run(self, arrive):
        from Attention import narrowing_view, narrow_step, interpretation_at, record_interpretation
        from ModelAttention import handoff
        state = self.walk.state
        slot = torch.zeros_like(self.present,dtype=torch.long)
        rows = self.present
        first = True
        while True:
            view = narrowing_view(state)
            interval = view[0].intervals[state.rows,slot]
            field = (self.model._sentence_open_fields[self.sid] if first else
                     perceive(self.model,interval,rows))
            self.fields.append(dict(field=field,where=interval.detach().clone(),active=rows.clone()))
            index, allowed = interpretation_at(state,slot)
            allowed &= rows
            options = {}
            turn = len(self.decisions)
            if self.prior is not None and turn < len(self.prior.decisions):
                old = self.prior.decisions[turn]
                if self.kept is not None:
                    replay = self.present
                else:
                    # Walk departures replay the prefix's grammar decisions;
                    # a compose departure keeps the walk's interpretations.
                    at = self.draw.get('interpret_round',torch.full_like(index,-1))
                    replay = ((self.draw['attention_round']<0) | (len(state.actions)<=self.draw['attention_round'])) & ((at<0)|(turn<at))
                    options['masked_action']=torch.where(at==turn,old['action'],-1)
                options['replay_action']=torch.where(replay,old['action'],-1)
            choice = state.chooser.choose_interpret(state.keys[state.rows,index],allowed,**options)
            choice.update(index=index,slot=slot.clone())
            self.decisions.append(choice)
            record_interpretation(state,slot,choice['interpreted'])
            handoff(self.model,self.reading())
            self.model._sentence_look_fields = dict(getattr(self.model,'_sentence_look_fields',{}))
            self.model._sentence_look_fields[self.sid] = tuple(self.fields)
            # Batch rows can interpret different words at the same look.
            for word in index[choice['interpreted']].unique().tolist():
                active = choice['interpreted'] & (index==word)
                arrive(torch.tensor(word,device=index.device),active)
            first = False
            if len(state.actions)>=int(state.budget) or not bool(narrowing_view(state)[8].any()):
                break
            narrow_step(state)
            slot,rows=state.look_slot,state.look_active & self.present
        return self.reading()
