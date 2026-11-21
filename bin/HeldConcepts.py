"""One bounded copy of the concepts held beside a sentence's grammatical tree.

Refresh and publication addresses come from the read, never a replayed choice.
The values are evaluated anew; no grammatical result from the read is copied.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ArrivalReadings:
    """Consumed candidates for MTM identity refinement, local to one trial.

    Keys are temporary constituent identities, not persistent concept IDs.
    Only the native arrival menu's readings are held; no operator tree is.
    """
    values: dict


@dataclass(frozen=True)
class Concepts:
    values: torch.Tensor
    identities: torch.Tensor
    timelines: torch.Tensor
    where: torch.Tensor
    when: torch.Tensor
    valid: torch.Tensor
    written: torch.Tensor

    def for_stream(self, stream):
        return type(self)(*(value[stream].clone() for value in self.__dict__.values()))

    def detached(self):
        return type(self)(*(value.detach().clone() for value in self.__dict__.values()))


def snapshot(state):
    from Memory import STMChunks as C
    valid = (state[C.STACK] < 0) & (state[C.IDENTITY] >= 0)
    return Concepts(*(state[column].clone() for column in
        (C.BUFFER,C.IDENTITY,C.TIMELINE,C.WHERE,C.WHEN)), valid, torch.zeros_like(valid))


class Evaluation:
    def __init__(self, initial, model):
        self.model = model
        self.concepts = (None if initial is None else
                      Concepts(*(value.clone() for value in initial.__dict__.values())))
        self.arrivals = None

    def retain_arrivals(self, clause, rules, idea_points):
        from ReferenceParticipants import arrival_members
        candidates, leaves, terms = arrival_members(clause, rules)
        offered = {id(node.composition): node.composition for node in candidates
                   if node.composition is not None}
        offered.update((key, terms[key]) for key in leaves)
        self.arrivals = ArrivalReadings({key: term.point for key, term in offered.items()
                                        if key not in idea_points})
        return self.arrivals

    def read(self, witness, *, relative=False):
        index, identity, timeline = map(int, witness)
        if index < 0:
            return None
        concepts = self.concepts
        if (concepts is None or index >= len(concepts.values) or not bool(concepts.valid[index])
                or int(concepts.identities[index]) != identity or int(concepts.timelines[index]) != timeline):
            raise ValueError(f'held-concept refresh reads outside the initial/published concepts: seat {index}, identity {identity}, timeline {timeline}')
        value = concepts.values[index]
        if relative:
            from RowTime import compared
            value = compared(self.model,value)
        return value

    def publish(self, witness, operands, result, frames, head):
        from ChunkJournal import publication_value
        index, identity, timeline, found, kind, earlier = map(int, witness)
        if index < 0:
            return
        concepts = self.concepts
        if concepts is None or index >= len(concepts.values):
            raise ValueError('held-concept publication has no constructed initial field')
        if found:
            self.read((index,identity,timeline))
        prior = frames[earlier,0] if kind == 1 else result
        value = publication_value(kind, operands, result, prior)
        where = concepts.where[index] if found else head.where
        when = concepts.when[index] if found else head.when
        band = self.model.languageSpace.chunk_journal.percept_band
        if band is not None and len(value) == band[2]:
            value = torch.cat((value[:band[0]],where,when,value[band[1]:]))
        values = concepts.values.clone(); values[index] = value
        ids, times = concepts.identities.clone(), concepts.timelines.clone()
        ids[index],times[index] = identity,timeline
        locations,stamps = concepts.where.clone(),concepts.when.clone()
        if where is not None: locations[index] = where
        if when is not None: stamps[index] = when
        valid,written = concepts.valid.clone(),concepts.written.clone()
        valid[index] = written[index] = True
        self.concepts = Concepts(values,ids,times,locations,stamps,valid,written)

    def identity_values(self, identities, timelines, original):
        if self.concepts is None or identities is None:
            return original
        values = original.clone()
        for row,(identity,timeline) in enumerate(zip(identities.tolist(),timelines)):
            match = (self.concepts.valid & (self.concepts.identities == identity)
                     & (self.concepts.timelines == timeline))
            if bool(match.any()):
                values[row] = self.concepts.values[int(match.long().argmax())]
        return values


def install(state, programs):
    """Before the kept close, replace only surviving off-stack publications."""
    from Memory import STMChunks as C
    values = state[C.BUFFER].clone()
    for stream,program in enumerate(programs):
        concepts = None if program is None else program.held_final
        if concepts is None:
            continue
        for seat in concepts.written.nonzero().flatten().tolist():
            match = ((state[C.STACK][stream] < 0)
                & (state[C.IDENTITY][stream] == concepts.identities[seat])
                & (state[C.TIMELINE][stream] == concepts.timelines[seat]))
            values[stream] = torch.where(match[:,None],concepts.values[seat].to(values),values[stream])
    return (values,*state[1:])
