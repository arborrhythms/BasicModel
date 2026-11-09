"""STM, MTM and LTM share one commit boundary: the kept close.

Trial-local STM updates are functional; a discarded trial changes no retained
store. The kept close retains STM's working set, commits MTM's identities and
writes LTM's durable ideas and relations. State keys and layouts do not depend
on this module path. STMChunks keeps its own reference tags, which differ from
LTM1's tags. ShortTermMemory still owns its unchanged shift/reduce scorer; that
coupling is carried for a later round.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import torch
from torch import nn

OCCURRENCE, SYMBOL, RELATION = 1, 2, 3
VERSION = 5
# Reserved outside both address inventories. It never consumes a learnable seat.
NULL_CONCEPT = torch.iinfo(torch.int64).max


from Layers import Layer, MeaningExpectation
from util import TheXMLConfig
from Meaning import ConceptualMeaning
from ClauseRow import ClauseRows
from Occurrence import OccurrenceRows, AddressIndex
from EntryRead import EntryRead

class STMChunks:
    """Functional operations over the fixed-capacity sentence working set."""
    BUFFER, DEPTH, ORDERS, GRAMMAR, ROWS, ACTIVATIONS = range(6)
    REFS, IDENTITY, TIMELINE, STACK, READ, WHERE, WHEN = range(6, 13)
    TRUTH = 13
    SYMBOL, IDENTITY_REF, ACTION, RELATION, OCCURRENCE = range(1, 6)
    FIELDS = ('buffer', 'depth', 'orders', 'grammar_orders', 'concept_rows',
              'concept_activations', 'chunk_refs', 'chunk_identity', 'chunk_timeline',
              'chunk_stack', 'chunk_read', 'chunk_where', 'chunk_when', 'chunk_truth')


    @staticmethod
    def empty(batch, capacity, width, *, device=None, dtype=torch.float32):
        """Construction only: B streams × stmCapacity chunks × conceptual width.

        Each chunk has two tagged references and two four-coordinate percept
        bands. Neither a word count nor a journal length can enlarge this set.
        """
        shape = (int(batch), int(capacity))
        if min(shape) < 1 or int(width) < 0:
            raise ValueError('STM construction requires positive streams and capacity')
        def longs(extra=(), fill=0):
            return torch.full((*shape, *extra), fill, dtype=torch.long, device=device)
        def floats(extra=()):
            return torch.zeros((*shape, *extra), dtype=dtype, device=device)
        return (floats((int(width),)), torch.zeros(shape[0], dtype=torch.long, device=device),
                longs(fill=-1), longs(fill=-1), longs(fill=-1), floats(),
                longs((2, 2)), longs(fill=-1), longs(), longs(fill=-1), longs(),
                floats((4,)), floats((4,)), floats((2,)))


    @staticmethod
    def locate(state, where, *, band=WHERE):
        """Resolve an occupied chunk by an exact where or when band."""
        if band not in (STMChunks.WHERE, STMChunks.WHEN):
            raise ValueError('chunk lookup requires the WHERE or WHEN band')
        where = torch.as_tensor(where, device=state[band].device, dtype=state[band].dtype)
        where = where.expand(state[band].shape[0], 4)
        match = (state[band] == where[:, None]).all(-1) & (state[STMChunks.READ] > 0)
        return torch.where(match.any(-1), match.long().argmax(-1), -1)


    @staticmethod
    def gather(value, indices):
        index = indices.reshape(*indices.shape, *([1] * (value.ndim-2)))
        return value.gather(1, index.expand(*indices.shape, *value.shape[2:]))


    @staticmethod
    def at(state, position=0):
        """Physical index for a newest-first stack position; -1 when absent."""
        position = torch.as_tensor(position, device=state[STMChunks.STACK].device)
        if position.ndim == 0:
            position = position.expand(state[STMChunks.STACK].shape[0])
        match = state[STMChunks.STACK] == position[:, None]
        return torch.where(match.any(-1), match.long().argmax(-1), -1)


    @staticmethod
    def stack_indices(state):
        positions = torch.arange(state[STMChunks.STACK].shape[1], device=state[STMChunks.STACK].device)
        match = state[STMChunks.STACK][:, None, :] == positions[None, :, None]
        return torch.where(match.any(-1), match.long().argmax(-1), -1)


    @staticmethod
    def stack_view(state):
        """A read-only positional view for grammar and legacy stack consumers."""
        indices = STMChunks.stack_indices(state)
        result = []
        for field, value in enumerate(state[:6]):
            if field == STMChunks.DEPTH:
                result.append(value)
                continue
            member = STMChunks.gather(value, indices.clamp_min(0))
            mask = (indices >= 0).reshape(*indices.shape, *([1]*(value.ndim-2)))
            result.append(torch.where(mask, member, -1 if field in (STMChunks.ORDERS, STMChunks.GRAMMAR, STMChunks.ROWS) else 0))
        return tuple(result)


    @staticmethod
    def write(value, indices, member, gate):
        positions = torch.arange(value.shape[1], device=value.device)[None]
        mask = gate.reshape(-1, 1) & (indices[:, None] >= 0) & (positions == indices[:, None])
        return torch.where(mask.reshape(*mask.shape, *([1]*(value.ndim-2))), member[:, None], value)


    @staticmethod
    def clear(state, mask):
        result = list(state)
        for field, value in enumerate(state):
            if field == STMChunks.DEPTH:
                continue
            selected = mask.reshape(*mask.shape, *([1]*(value.ndim-2)))
            result[field] = torch.where(selected, -1 if field in (STMChunks.ORDERS, STMChunks.GRAMMAR, STMChunks.ROWS, STMChunks.IDENTITY, STMChunks.STACK) else 0, value)
        result[STMChunks.DEPTH] = (result[STMChunks.STACK] >= 0).sum(-1)
        return tuple(result)


    @staticmethod
    def allocate_chunk(state, gate):
        """Return (cleared state, index); all-stack streams refuse with -1."""
        off = state[STMChunks.STACK] < 0
        free = off & (state[STMChunks.READ] == 0)
        oldest = state[STMChunks.READ].masked_fill(~off, torch.iinfo(torch.long).max).argmin(-1)
        index = torch.where(free.any(-1), free.long().argmax(-1), oldest)
        valid = gate.reshape(-1).bool() & off.any(-1)
        index = torch.where(valid, index, -1)
        mask = torch.arange(off.shape[1], device=off.device)[None] == index[:, None]
        return STMChunks.clear(state, mask), index


    @staticmethod
    def read(state, indices, gate=None, *, tick=None):
        """Refresh only actual reads; enumerating an offered bind calls no read."""
        if indices.ndim == 1:
            indices = indices[:, None]
        if gate is None:
            gate = torch.ones_like(indices, dtype=torch.bool)
        elif gate.ndim == 1:
            gate = gate[:, None]
        tick = state[STMChunks.READ].amax(-1)+1 if tick is None else torch.as_tensor(tick, device=indices.device).expand(indices.shape[0])
        positions = torch.arange(state[STMChunks.STACK].shape[1], device=indices.device)[None, :, None]
        touched = ((positions == indices[:, None]) & gate[:, None] & (indices[:, None] >= 0)).any(-1)
        result = list(state)
        result[STMChunks.READ] = torch.where(touched, tick[:, None], state[STMChunks.READ])
        return tuple(result)


    @staticmethod
    def deposit(state, value, gate, *, order=None, grammar_order=None, row=None,
                activation=None, refs=None, identity=None, timeline=None,
                where=None, when=None, truth=None, stack=False, tick=None):
        """Allocate one arrival; remembered and prepended entries stay off stack."""
        before = state
        state, index = STMChunks.allocate_chunk(state, gate)
        active = index >= 0
        result = list(state)
        b = value.shape[0]
        defaults = {STMChunks.BUFFER:value, STMChunks.ORDERS:order, STMChunks.GRAMMAR:grammar_order, STMChunks.ROWS:row,
                    STMChunks.ACTIVATIONS:activation, STMChunks.REFS:refs, STMChunks.IDENTITY:identity,
                    STMChunks.TIMELINE:timeline, STMChunks.WHERE:where, STMChunks.WHEN:when, STMChunks.TRUTH:truth}
        for field, member in defaults.items():
            if member is None:
                continue
            member = torch.as_tensor(member, device=state[field].device, dtype=state[field].dtype)
            member = member.expand(b, *state[field].shape[2:])
            result[field] = STMChunks.write(state[field], index, member, active)
        if stack:
            result[STMChunks.STACK] = torch.where(active[:, None] & (state[STMChunks.STACK] >= 0), state[STMChunks.STACK]+1, state[STMChunks.STACK])
            result[STMChunks.STACK] = STMChunks.write(result[STMChunks.STACK], index, torch.zeros_like(index), active)
            result[STMChunks.DEPTH] = before[STMChunks.DEPTH] + active.long()
        next_tick = before[STMChunks.READ].amax(-1)+1 if tick is None else tick
        return STMChunks.read(tuple(result), index, active, tick=next_tick), index


    @staticmethod
    def resolve_top_reference(state, value, row, order, activation, gate):
        index = STMChunks.at(state)
        gate = gate.reshape(-1).bool() & (index >= 0) & (row.reshape(-1) >= 0)
        result = list(state)
        for field, member in ((STMChunks.BUFFER,value), (STMChunks.ROWS,row), (STMChunks.ORDERS,order), (STMChunks.ACTIVATIONS,activation)):
            result[field] = STMChunks.write(state[field], index, member.reshape(value.shape[0], *state[field].shape[2:]), gate)
        return tuple(result)


    @staticmethod
    def soft_reset_rows(state, gate):
        """Drop only unfinished sentence operands; keep recalled/recent chunks."""
        return STMChunks.clear(state, gate.reshape(-1, 1).bool() & (state[STMChunks.STACK] >= 0))


    @staticmethod
    def keep_close(state, gate, *, refs, value=None, when=None):
        """Retain the kept top as an off-stack recent entry, freeing its peers."""
        top = STMChunks.at(state)
        active = gate.reshape(-1).bool() & (top >= 0)
        positions = torch.arange(state[STMChunks.STACK].shape[1], device=top.device)[None]
        result = list(STMChunks.clear(state, active[:, None] & (state[STMChunks.STACK] >= 0) & (positions != top[:, None])))
        for field, member in ((STMChunks.STACK,torch.full_like(top,-1)), (STMChunks.REFS,refs), (STMChunks.BUFFER,value), (STMChunks.WHEN,when)):
            if member is not None:
                result[field] = STMChunks.write(result[field], top, member, active)
        result[STMChunks.DEPTH] = (result[STMChunks.STACK] >= 0).sum(-1)
        return STMChunks.read(tuple(result), top, active)


    @staticmethod
    def refresh(state, indices, gate=None):
        """Read the current value of a bound off-stack identity/timeline."""
        identities = STMChunks.gather(state[STMChunks.IDENTITY], indices.clamp_min(0))
        timelines = STMChunks.gather(state[STMChunks.TIMELINE], indices.clamp_min(0))
        match = ((identities[..., None] == state[STMChunks.IDENTITY][:, None])
                 & (timelines[..., None] == state[STMChunks.TIMELINE][:, None])
                 & (state[STMChunks.STACK][:, None] < 0) & (identities[..., None] >= 0))
        field = match & ((state[STMChunks.REFS][:,None,:,0,0] == STMChunks.SYMBOL)
                         | (state[STMChunks.REFS][:,None,:,0,0] == STMChunks.IDENTITY_REF))
        match = torch.where(field.any(-1)[...,None],field,match)
        sources = match.long().argmax(-1)
        available = match.any(-1) & (indices >= 0)
        if gate is not None:
            available &= gate
        values = torch.where(available[..., None], STMChunks.gather(state[STMChunks.BUFFER], sources),
                             STMChunks.gather(state[STMChunks.BUFFER], indices.clamp_min(0)))
        state = STMChunks.read(state, torch.where(available, sources, -1), available)
        return STMChunks.read(state, indices, gate), values


    @staticmethod
    def bind_menu(state, mention, *, exclusions=None, meaning_start=0):
        """Fixed K+1 field variants: dummy first, then eligible off-stack chunks."""
        batch, positions, width = mention.shape
        own = mention[:, :, None]
        candidates = state[STMChunks.BUFFER][:, None].expand(batch, positions, -1, width)
        available = ((state[STMChunks.STACK] < 0) & (state[STMChunks.IDENTITY] >= 0))[:, None].expand(batch, positions, -1)
        from IdentityRows import meet_ratios
        covered, covers = meet_ratios(own[..., meaning_start:], candidates[..., meaning_start:])
        available = available & ((covered == 1) | (covers == 1))
        if exclusions is not None:
            x, y = exclusions[:, 0], exclusions[:, 1]
            ax, ay = own.index_select(-1, x) >= .3, own.index_select(-1, y) >= .3
            bx, by = candidates.index_select(-1, x) >= .3, candidates.index_select(-1, y) >= .3
            available = available & ~((ax & ~bx & by & ~ay) | (ay & ~by & bx & ~ax)).any(-1)
        dummy = torch.zeros(batch, positions, 1, dtype=torch.long, device=mention.device)
        chunks = torch.arange(state[STMChunks.STACK].shape[1], device=mention.device)[None, None].expand(batch, positions, -1)
        return (torch.cat((torch.zeros_like(own), candidates), 2),
                torch.cat((dummy, state[STMChunks.IDENTITY][:, None].expand_as(chunks)), 2),
                torch.cat((dummy-1, chunks), 2),
                torch.cat((torch.ones_like(dummy,dtype=torch.bool), available), 2))


    @staticmethod
    def apply_choice(state, choice, *, operand_actions=None):
        """Write the grammatical head's physical chunk and free only a consumed operand."""
        active = choice.applied.bool()
        bind = (torch.zeros_like(active) if choice.stack_arity is None else
                (choice.kind == 1) & (choice.stack_arity == 1))
        binary = active & (choice.kind == 1) & ~bind
        unary = active & ((choice.kind == 2) | bind)
        newest, older = STMChunks.at(state), STMChunks.at(state, 1)
        chosen = STMChunks.at(state, choice.position)
        head = torch.ones_like(newest) if choice.head_role is None else choice.head_role
        target = torch.where(binary, torch.where(head == 2, newest, older),
                             torch.where(bind, newest, chosen))
        other = torch.where(head == 2, older, newest)
        valid = (binary | unary) & (target >= 0)
        positions = torch.arange(state[STMChunks.STACK].shape[1], device=newest.device)[None]
        state = STMChunks.read(state, torch.stack((newest, torch.where(binary, older, chosen)), -1),
                     torch.stack((active, binary | unary), -1))
        result = list(STMChunks.clear(state, binary[:, None] & (positions == other[:, None])))
        result[STMChunks.STACK] = torch.where(binary[:, None] & (result[STMChunks.STACK] >= 2), result[STMChunks.STACK]-1, result[STMChunks.STACK])
        result[STMChunks.STACK] = STMChunks.write(result[STMChunks.STACK], target, torch.zeros_like(target), binary)
        result[STMChunks.DEPTH] = state[STMChunks.DEPTH]-binary.long()
        result[STMChunks.BUFFER] = STMChunks.write(result[STMChunks.BUFFER], target, choice.candidate, valid)
        previous = STMChunks.gather(state[STMChunks.GRAMMAR], target.clamp_min(0)[:, None])[:, 0]
        pair = torch.stack((STMChunks.gather(state[STMChunks.GRAMMAR], newest.clamp_min(0)[:, None])[:, 0],
                            STMChunks.gather(state[STMChunks.GRAMMAR], older.clamp_min(0)[:, None])[:, 0]), -1)
        grammar = torch.where(binary, torch.where((pair >= 0).all(-1), pair.amax(-1)+1, -1),
                              torch.where(previous >= 0, previous+1, -1))
        for field, member in ((STMChunks.GRAMMAR,grammar), (STMChunks.ORDERS,choice.order),
                               (STMChunks.ROWS,torch.full_like(newest,-1)), (STMChunks.ACTIVATIONS,choice.candidate.new_zeros(len(newest)))):
            if member is not None:
                result[field] = STMChunks.write(result[field], target, member, valid)
        # SYMBOL/IDENTITY and reference-free compressed values are resolved.
        # ACTION records remain only until both operands can be compressed.
        unresolved = (state[STMChunks.REFS][..., 0] == STMChunks.ACTION).any(-1)
        unresolved = STMChunks.gather(unresolved, torch.stack((older, newest), -1).clamp_min(0)).any(-1)
        references = state[STMChunks.REFS].new_zeros(len(newest), 2, 2)
        if operand_actions is not None:
            references = torch.stack((torch.full_like(operand_actions,STMChunks.ACTION), operand_actions), -1)
            references = torch.where(unresolved[:, None, None], references, 0)
        result[STMChunks.REFS] = STMChunks.write(result[STMChunks.REFS], target, references, binary)
        if choice.identity is not None:
            result[STMChunks.IDENTITY] = STMChunks.write(result[STMChunks.IDENTITY], target, choice.identity, bind & valid)
        if choice.chunk is not None:
            selected = choice.chunk.clamp_min(0)
            timeline = STMChunks.gather(state[STMChunks.TIMELINE], selected[:, None])[:, 0]
            result[STMChunks.TIMELINE] = STMChunks.write(result[STMChunks.TIMELINE], target, timeline, bind & valid & (choice.chunk >= 0))
            result = list(STMChunks.read(tuple(result), choice.chunk, bind & valid & (choice.chunk >= 0)))
        # Transpose exchanges stack positions, never the chunks' values or ages.
        transpose = active & (choice.kind == 3)
        result[STMChunks.STACK] = torch.where(transpose[:, None] & (state[STMChunks.STACK] >= 0) & (state[STMChunks.STACK] < 2),
                                    1-state[STMChunks.STACK], result[STMChunks.STACK])
        return tuple(result)


class MemoryStore:
    """Shared tensor ownership, without adding state or a storage layout.

    A snapshot owns its storage. Committed snapshots detach the graph; STM's
    trial snapshots may retain it. CPU checkpoint copies use the same rule.
    """

    @staticmethod
    def _snapshot_tensor(value, *, detach=True, cpu=False):
        value = value.detach() if detach else value
        if cpu:
            value = value.cpu()
        return value.clone()


class ShortTermMemory(MemoryStore, Layer):
    """Runtime idea stack plus shift/reduce scorer for ConceptualSpace.

    Newest ideas live at slot 0; capacity defaults to 8 and is configurable
    via ``<ConceptualSpace><stmCapacity>``. See ``doc/STM.md``.
    """

    DEFAULT_CAPACITY = 8

    def __init__(self, batch=1, capacity=None, concept_dim=0):
        super().__init__(nInput=0, nOutput=0)
        # Construction bound: nStreams × stmCapacity (default eight) × D.
        # The same fixed tensors serve parsing, recent entries and recall.
        self._init_capacity = int(capacity or self.DEFAULT_CAPACITY)
        self._init_concept_dim = int(concept_dim)
        self._init_batch = self._active_batch = int(batch)
        self._word_subspace = None
        for name, value in zip(STMChunks.FIELDS, STMChunks.empty(
                batch, self._init_capacity, concept_dim)):
            object.__setattr__(self, '_live_'+name, value)
        self._live_capacity = self._init_capacity
        self._live_max_depth_host = 0
        # Scope reserves 2 clause flags/references + 12 journal head fields;
        # whole provenance reserves 3 indices + 2 span endpoints. Prefix
        # views may expose the legacy widths, never shrink these owners.
        from ChunkJournal import empty_scope
        self._scope_columns, self._whole_columns = 2, 3
        self._live_clause_state = empty_scope(self._live_buffer)
        self._live_wholes = torch.full((batch, self._init_capacity, 5), -1, dtype=torch.long)
        # One unary-history entry per constructed chunk and depth per stream.
        self._live_last_unary = torch.full((batch, self._init_capacity), -1, dtype=torch.long)
        self._live_unary_history_depth = torch.zeros(batch,dtype=torch.long)
        self._thought_contents = tuple((None,) * self._init_capacity for _ in range(batch))
        self._thought_clock = (0,) * self._init_batch

    @property
    def state(self):
        return tuple(getattr(self, '_live_'+name)[:self._active_batch] for name in STMChunks.FIELDS)

    def set_state(self, state):
        if len(state) != len(STMChunks.FIELDS):
            raise ValueError('STM requires the complete shared chunk state')
        batch = state[0].shape[0]
        self.ensure_batch(batch)
        for name, value in zip(STMChunks.FIELDS, state):
            previous = getattr(self, '_live_'+name)
            if value.shape[1:] != previous.shape[1:]:
                raise ValueError('STM chunk shape exceeds its construction')
            full = value if batch == self._init_batch else torch.cat((value, previous[batch:].to(value)), 0)
            if name == 'chunk_truth' and full.device == previous.device and full.dtype == previous.dtype:
                with torch.no_grad():
                    previous.copy_(full)
            else:
                self._assign_live(self, '_live_'+name, full)
        if not torch.compiler.is_compiling():
            self._max_depth_host = int(state[1].max())

    def attach_word_subspace(self, word_subspace):
        if int(word_subspace._idea_capacity) != self._init_capacity or int(word_subspace._stm_payload_dim) != self._init_concept_dim:
            raise ValueError('attached STM owner disagrees with the constructed chunk shape')
        object.__setattr__(self, '_word_subspace', word_subspace)

    # -- per-forward live working buffer ---------------------------------
    # Fresh, plain-attribute tensors avoid torch.compile input mutation.

    @staticmethod
    def _assign_live(owner, name, value):
        """Install a live STM tensor.

        Under the compiler this is an in-place copy whenever the existing
        tensor has the same shape/dtype/device: with a ``torch.while_loop``
        in the graph, dynamo (torch 2.14 nightly) drops a later attribute
        *assignment* of an attribute assigned earlier in the same graph
        (the per-forward seed), so the reads after the word loop -- the
        sentence reduce and the reconstruction loss -- saw the empty seed.
        An in-place copy is tracked correctly and keeps the gradient.
        Eager execution keeps the plain assignment (the buffer may change
        shape, and a persisted buffer must not be mutated in place).
        """
        current = owner.__dict__.get(name)
        if (torch.compiler.is_compiling() and torch.is_tensor(value)
                and torch.is_tensor(current)
                and tuple(current.shape) == tuple(value.shape)
                and current.dtype == value.dtype
                and current.device == value.device
                and current is not value):
            current.copy_(value)
            return
        object.__setattr__(owner, name, value)

    def detach_live(self):
        """Cut the autograd history of the live STM state at a brick
        boundary.  The state carries values across bricks; under the
        compiler it is updated by in-place copies (``_assign_live``), and
        an in-place copy on a tensor with a graph chains every brick's
        graph to the previous one (the loop checkpoints of all earlier
        bricks stayed alive: 1-1.5 GiB per brick at B = 8, W = 256)."""
        for name, value in list(self.__dict__.items()):
            if name.startswith("_live_") and torch.is_tensor(value) \
                    and value.grad_fn is not None:
                object.__setattr__(self, name, value.detach())

    @property
    def _buffer(self):
        return self._live_buffer[:self._active_batch]

    @_buffer.setter
    def _buffer(self, value):
        if value.shape != self._buffer.shape:
            raise ValueError('STM buffer cannot resize after construction')
        full = value if self._active_batch == self._init_batch else torch.cat((value, self._live_buffer[self._active_batch:].to(value)), 0)
        self._assign_live(self, '_live_buffer', full)

    @property
    def _depth(self):
        return self._live_depth[:self._active_batch]

    @_depth.setter
    def _depth(self, value):
        self._set_member('depth', value)

    @property
    def _orders(self):
        return self._live_orders[:self._active_batch]

    @_orders.setter
    def _orders(self, value):
        self._set_member('orders', value)

    @property
    def _grammar_orders(self):
        return self._live_grammar_orders[:self._active_batch]

    @_grammar_orders.setter
    def _grammar_orders(self, value):
        self._set_member('grammar_orders', value)

    @property
    def _concept_rows(self):
        return self._live_concept_rows[:self._active_batch]

    @_concept_rows.setter
    def _concept_rows(self, value):
        self._set_member('concept_rows', value)

    @property
    def _concept_activations(self):
        return self._live_concept_activations[:self._active_batch]

    @_concept_activations.setter
    def _concept_activations(self, value):
        self._set_member('concept_activations', value)

    def _set_member(self, name, value):
        previous = getattr(self, '_live_'+name)
        if value.shape != previous[:self._active_batch].shape:
            raise ValueError('STM metadata cannot resize after construction')
        full = value if self._active_batch == self._init_batch else torch.cat(
            (value, previous[self._active_batch:].to(value)), 0)
        self._assign_live(self, '_live_'+name, full)

    @property
    def _clause_state(self):
        return self._live_clause_state[:self._active_batch,:,:self._scope_columns]

    @_clause_state.setter
    def _clause_state(self, value):
        if value.ndim != 3 or value.shape[-1] not in (2,14):
            raise ValueError('STM clause scope exceeds its constructed provenance width')
        columns = value.shape[-1]
        padded = torch.nn.functional.pad(value,(0,14-columns))
        self._set_member('clause_state',padded)
        self._scope_columns = columns

    @property
    def _last_unary(self):
        return self._live_last_unary[:self._active_batch]

    @_last_unary.setter
    def _last_unary(self, value):
        self._set_member('last_unary',value)

    @property
    def _unary_history_depth(self):
        return self._live_unary_history_depth[:self._active_batch]

    @_unary_history_depth.setter
    def _unary_history_depth(self, value):
        self._set_member('unary_history_depth',value)

    @property
    def _max_depth_host(self):
        return self._live_max_depth_host

    @_max_depth_host.setter
    def _max_depth_host(self, value):
        self._live_max_depth_host = int(value)

    @property
    def capacity(self):
        ss = self._word_subspace
        if ss is None:
            return self._live_capacity
        return ss._idea_capacity

    @property
    def concept_dim(self):
        ss = self._word_subspace
        if ss is None:
            return self._init_concept_dim
        return ss._stm_payload_dim

    def begin_forward(self, batch, device=None, dtype=None):
        """Start another forward in the same constructed shared working set."""
        self.ensure_batch(batch)
        for name in STMChunks.FIELDS:
            value = getattr(self, '_live_'+name)
            target = dtype if value.is_floating_point() and dtype is not None else value.dtype
            self._assign_live(self, '_live_'+name, value.detach().to(device=device, dtype=target))
        gate = torch.ones(batch, dtype=torch.bool, device=self._buffer.device)
        self.set_state(STMChunks.soft_reset_rows(self.state, gate))
        from ClauseScope import ClauseScope
        self._clause_state = ClauseScope.reset(self._clause_state[:batch].to(self._buffer.device), gate)

    def ensure_clause_state(self):
        from ClauseScope import ClauseScope
        value = getattr(self, '_clause_state', None)
        if not torch.is_tensor(value) or value.shape[:2] != self._buffer.shape[:2]:
            raise ValueError('STM clause scope disagrees with its construction')
        if value.device != self._buffer.device:
            value = value.to(self._buffer.device)
            self._clause_state = value
        return value

    def ensure_order_state(self):
        for slab in (self._orders, self._grammar_orders):
            if slab.shape != self._buffer.shape[:2] or slab.device != self._buffer.device:
                raise ValueError('STM order state disagrees with its construction')
        return self._orders, self._grammar_orders

    def ensure_reference_state(self):
        for slab in (self._concept_rows, self._concept_activations):
            if slab.shape != self._buffer.shape[:2] or slab.device != self._buffer.device:
                raise ValueError('STM reference state disagrees with its construction')
        return self._concept_rows, self._concept_activations

    @staticmethod
    def _coerce_order_rows(value, batch, device, *, default=-1):
        if value is None:
            return torch.full(
                (int(batch),), int(default), dtype=torch.long, device=device)
        if torch.is_tensor(value):
            return value.to(device=device, dtype=torch.long).reshape(int(batch))
        return torch.full(
            (int(batch),), int(value), dtype=torch.long, device=device)

    @staticmethod
    def _coerce_concept_rows(value, batch, device):
        if value is None:
            return torch.full(
                (int(batch),), -1, dtype=torch.long, device=device)
        if torch.is_tensor(value):
            return value.to(
                device=device, dtype=torch.long).reshape(int(batch))
        return torch.full(
            (int(batch),), int(value), dtype=torch.long, device=device)

    @staticmethod
    def _coerce_concept_activations(value, batch, device, dtype,
                                    concept_rows=None):
        if value is None:
            activations = torch.zeros(
                (int(batch),), dtype=dtype, device=device)
        elif torch.is_tensor(value):
            activations = value.to(
                device=device, dtype=dtype).reshape(int(batch))
        else:
            activations = torch.full(
                (int(batch),), float(value), dtype=dtype, device=device)
        if concept_rows is not None:
            activations = torch.where(
                concept_rows >= 0, activations,
                torch.zeros_like(activations))
        return activations

    # -- idea-stack API (chart compose consumer surface) ------------------
    #
    # A5: all operations act on the single live store (the ``_buffer`` /
    # ``_depth`` / ``_max_depth_host`` properties -> the ``_live_*`` plain
    # attributes). The former ``ss is None`` branch (SymbolSubSpace-attached
    # vs standalone ``_fallback_*``) is gone -- the live data is no longer a
    # registered buffer on either, so the compiled forward never sees it as a
    # graph input. ``capacity`` still proxies to ``ss._idea_capacity`` when
    # attached so an externally-grown idea-stack capacity sizes the next
    # ``begin_forward`` correctly.

    def ensure_batch(self, batch):
        if not 0 < int(batch) <= self._init_batch:
            raise ValueError('STM batch exceeds its constructed nStreams')
        self._active_batch = int(batch)

    # -- slot-whole provenance (meronomy fold-ladder plan, Phase 2b) --------
    # One fixed-shape slab ``_wholes`` ``[B, capacity, 3]`` of longs mirrors
    # the buffer discipline (newest at slot 0): per slot the index of the
    # coarser analysis whole (the word) the unit belongs to, the unit's loop
    # position and the clause whole's index; ``-1`` = none / a fold across
    # wholes.  It is carried through the compiled word loop like the
    # reference slabs, so the pure ``functional_wholes_*`` primitives below
    # are the only mutation paths (eager methods wrap them).

    @property
    def _wholes(self):
        return self._live_wholes[:self._active_batch,:,:self._whole_columns]

    @_wholes.setter
    def _wholes(self, value):
        if value.ndim != 3 or value.shape[-1] not in (3,5):
            raise ValueError('STM whole provenance exceeds its constructed width')
        columns = value.shape[-1]
        self._set_member('wholes',torch.nn.functional.pad(value,(0,5-columns),value=-1))
        self._whole_columns = columns

    def ensure_whole_state(self):
        """Return the active view of the constructed whole/span provenance."""
        buf = self._buffer
        wholes = getattr(self, "_wholes", None)
        if (not torch.is_tensor(wholes) or wholes.shape[:2] != buf.shape[:2]
                or wholes.dtype != torch.long):
            raise ValueError('STM whole provenance disagrees with its construction')
        if wholes.device != buf.device:
            wholes = wholes.to(buf.device)
            self._wholes = wholes
        return wholes

    @staticmethod
    def functional_wholes_push(wholes, gate_b, whole_b, unit_b, clause_b):
        """Masked newest-first push of ``(whole, unit, clause)`` at slot 0."""
        B, cap, _ = wholes.shape
        gate = gate_b.reshape(B).to(device=wholes.device, dtype=torch.bool)
        entry = torch.stack(
            (whole_b.reshape(B).to(dtype=torch.long, device=wholes.device),
             unit_b.reshape(B).to(dtype=torch.long, device=wholes.device),
             clause_b.reshape(B).to(dtype=torch.long, device=wholes.device)),
            dim=-1).reshape(B, 1, 3)
        if wholes.shape[-1] == 5:
            unit = unit_b.reshape(B, 1, 1)
            entry = torch.cat((entry, unit, unit+1), -1)
        shifted = torch.cat((entry, wholes[:, :cap - 1]), dim=1)
        return torch.where(gate.reshape(B, 1, 1), shifted, wholes)

    @staticmethod
    def functional_wholes_reduce(wholes, applied_b):
        """Fold the two newest slots: the parent keeps a whole (and a clause)
        when both operands shared it, else ``-1``; a fold is no single
        unit.  Rows where ``applied_b`` is off are bit-identical."""
        B, cap, _ = wholes.shape
        if int(cap) < 2:
            return wholes
        gate = applied_b.reshape(B).to(
            device=wholes.device, dtype=torch.bool)
        newer, older = wholes[:, 0], wholes[:, 1]
        same_w = torch.logical_and(newer[:, 0] >= 0, newer[:, 0] == older[:, 0])
        same_c = torch.logical_and(newer[:, 2] >= 0, newer[:, 2] == older[:, 2])
        minus = torch.full_like(newer[:, 0], -1)
        parent = torch.stack(
            (torch.where(same_w, newer[:, 0], minus), minus,
             torch.where(same_c, newer[:, 2], minus)), dim=-1).reshape(B, 1, 3)
        if wholes.shape[-1] == 5:
            low = torch.minimum(newer[:, 3], older[:, 3])
            high = torch.maximum(newer[:, 4], older[:, 4])
            parent = torch.cat((parent, torch.stack((low, high), -1)[:, None]), -1)
        shifted = torch.cat(
            (parent, wholes[:, 2:], torch.full_like(wholes[:, :1], -1)), dim=1)
        return torch.where(gate.reshape(B, 1, 1), shifted, wholes)

    @staticmethod
    def functional_wholes_reset(wholes, gate_b):
        """Clear the provenance of the gated rows (a sentence boundary)."""
        B = int(wholes.shape[0])
        gate = gate_b.reshape(B).to(device=wholes.device, dtype=torch.bool)
        return torch.where(
            gate.reshape(B, 1, 1), torch.full_like(wholes, -1), wholes)

    @staticmethod
    def same_whole(wholes):
        """``[B]`` bool: the two newest slots share a coarser whole at some
        rung above the unit (the same word, or the same clause)."""
        if int(wholes.shape[1]) < 2:
            return torch.zeros(
                int(wholes.shape[0]), dtype=torch.bool, device=wholes.device)
        newer, older = wholes[:, 0], wholes[:, 1]
        same_w = torch.logical_and(newer[:, 0] >= 0, newer[:, 0] == older[:, 0])
        same_c = torch.logical_and(newer[:, 2] >= 0, newer[:, 2] == older[:, 2])
        return torch.logical_or(same_w, same_c)

    @staticmethod
    def shared_whole(wholes):
        """``[B]`` long: the whole the two newest slots share (the word, else
        the clause), ``-1`` when none."""
        B = int(wholes.shape[0])
        if int(wholes.shape[1]) < 2:
            return torch.full((B,), -1, dtype=torch.long, device=wholes.device)
        newer, older = wholes[:, 0], wholes[:, 1]
        same_w = torch.logical_and(newer[:, 0] >= 0, newer[:, 0] == older[:, 0])
        same_c = torch.logical_and(newer[:, 2] >= 0, newer[:, 2] == older[:, 2])
        minus = torch.full_like(newer[:, 0], -1)
        return torch.where(
            same_w, newer[:, 0], torch.where(same_c, newer[:, 2], minus))

    @staticmethod
    def newest_units(wholes):
        """``[B, 2]`` long: (unit of slot 1, unit of slot 0), ``-1`` unknown."""
        B = int(wholes.shape[0])
        if int(wholes.shape[1]) < 2:
            return torch.full((B, 2), -1, dtype=torch.long, device=wholes.device)
        return torch.stack((wholes[:, 1, 1], wholes[:, 0, 1]), dim=-1)

    def note_whole_masked(self, gate_rows, wholes, unit=-1, clauses=None):
        """Eager wrapper: record ``(whole, unit, clause)`` for a slot-0 push
        on the gated rows (tensors or lists; ``unit`` may be one int)."""
        slab = self.ensure_whole_state()
        B = int(slab.shape[0])
        gate = torch.as_tensor(gate_rows, device=slab.device).reshape(B)
        whole = torch.as_tensor(wholes, device=slab.device).reshape(B)
        unit_t = (torch.as_tensor(unit, device=slab.device).reshape(-1)
                  .expand(B) if torch.is_tensor(unit) or isinstance(unit, (list, tuple))
                  else torch.full((B,), int(unit), dtype=torch.long, device=slab.device))
        clause = (torch.as_tensor(clauses, device=slab.device).reshape(B)
                  if clauses is not None
                  else torch.full((B,), -1, dtype=torch.long, device=slab.device))
        self._wholes = self.functional_wholes_push(
            slab, gate, whole, unit_t, clause)

    def note_reduce_wholes(self, reduced_rows):
        """Eager wrapper of :meth:`functional_wholes_reduce`."""
        slab = self.ensure_whole_state()
        gate = torch.as_tensor(reduced_rows, device=slab.device)
        self._wholes = self.functional_wholes_reduce(slab, gate)

    def same_whole_rows(self):
        """Eager wrapper of :meth:`same_whole` on the live slab."""
        return self.same_whole(self.ensure_whole_state())


    def ensure_capacity(self, capacity):
        if int(capacity) > self._init_capacity:
            raise ValueError('STM cannot resize its constructed stmCapacity')

    def push(self, b, idea, order=None, grammar_order=None, concept_row=None, concept_activation=None):
        gate = torch.arange(self._active_batch, device=self._buffer.device) == int(b)
        if self.is_full(b):
            raise RuntimeError('ShortTermMemory.push: all chunks are on the stack; reduce before pushing')
        ideas = idea.to(self._buffer)[None].expand(self._active_batch, -1)
        self.push_step_masked(ideas, gate, order, grammar_order, concept_row, concept_activation)

    def push_step(self, ideas, orders=None, grammar_orders=None, concept_row=None, concept_activation=None):
        self.push_step_masked(ideas, torch.ones(ideas.shape[0], dtype=torch.bool, device=ideas.device),
                              orders, grammar_orders, concept_row, concept_activation)

    def push_step_masked(self, ideas, gate_b_1, orders=None, grammar_orders=None, concept_row=None, concept_activation=None):
        batch = ideas.shape[0]
        self.ensure_batch(batch)
        rows = self._coerce_concept_rows(concept_row, batch, ideas.device)
        state = self.functional_push_step_masked(self.state, ideas, gate_b_1,
            self._coerce_order_rows(orders, batch, ideas.device),
            self._coerce_order_rows(grammar_orders, batch, ideas.device), rows,
            self._coerce_concept_activations(concept_activation, batch, ideas.device, ideas.dtype, rows))
        moved = state[1] > self._depth
        self.set_state(state)
        from ClauseScope import ClauseScope
        self._clause_state = ClauseScope.push(self.ensure_clause_state(), moved, torch.full_like(rows,-1))

    @staticmethod
    def functional_push_step_masked(state, ideas, gate_b_1, inserted_orders, inserted_grammar_orders,
                                    inserted_concept_rows, inserted_concept_activations):
        return STMChunks.deposit(state, ideas, gate_b_1, stack=True,
            order=inserted_orders, grammar_order=inserted_grammar_orders,
            row=inserted_concept_rows, activation=inserted_concept_activations,
            truth=(1., 0.))[0]

    @staticmethod
    def functional_snapshot_mask(buffer, depth, active_rows=None):
        """Return the fixed-capacity STM snapshot and its occupied mask.

        Unlike :meth:`snapshot`, this operation never trims the slot axis
        using a Python high-water mark.  Consumers use the returned mask, so a
        single compiled graph supports every runtime sentence length and
        every independently progressing row in a large batch.
        """
        if buffer.dim() != 3 or depth.dim() != 1:
            raise ValueError(
                "functional STM snapshot expects [B,K,D] and depth [B]")
        B, cap = int(buffer.shape[0]), int(buffer.shape[1])
        if int(depth.shape[0]) != B:
            raise ValueError(
                "functional STM snapshot depth must contain one value per row")
        slots = torch.arange(
            cap, dtype=depth.dtype, device=depth.device).reshape(1, cap)
        occupied = slots < depth.reshape(B, 1)
        if active_rows is not None:
            occupied = torch.logical_and(
                occupied,
                active_rows.reshape(B, 1).to(
                    device=occupied.device, dtype=torch.bool))
        return buffer, occupied

    def push_window_batch(self, ideas, orders=None, grammar_orders=None, concept_rows=None, concept_activations=None):
        for i in range(ideas.shape[1]):
            def item(value):
                return value[:, i] if torch.is_tensor(value) else value
            self.push_step(ideas[:, i], item(orders), item(grammar_orders), item(concept_rows), item(concept_activations))

    def pop(self, b):
        index = STMChunks.at(self.state)
        if int(index[b]) < 0:
            return None
        idea = self._buffer[b, index[b]].clone()
        rows = torch.arange(self._active_batch, device=index.device) == int(b)
        positions = torch.arange(self.capacity, device=index.device)[None]
        state = list(STMChunks.clear(self.state, rows[:, None] & (positions == index[:, None])))
        state[STMChunks.STACK] = torch.where(rows[:, None] & (state[STMChunks.STACK] >= 0),
                                            state[STMChunks.STACK]-1, state[STMChunks.STACK])
        self.set_state(tuple(state))
        return idea

    def peek(self, b, n=0):
        indices = STMChunks.at(self.state, n)
        if int(indices[b]) < 0:
            return None
        gate = torch.arange(self._active_batch, device=indices.device) == int(b)
        self.set_state(STMChunks.read(self.state, indices, gate))
        return self._buffer[b, indices[b]]

    def snapshot(self, detach=False):
        width = min(int(self._max_depth_host), self.capacity)
        if width <= 0:
            return None
        value = STMChunks.stack_view(self.state)[0][:, :width]
        return self._snapshot_tensor(value, detach=detach)

    def snapshot_orders(self, detach=False):
        width = min(int(self._max_depth_host), self.capacity)
        if width <= 0:
            return None
        value = STMChunks.stack_view(self.state)[2][:, :width]
        return self._snapshot_tensor(value, detach=detach)

    def snapshot_grammar_orders(self, detach=False):
        width = min(int(self._max_depth_host), self.capacity)
        if width <= 0:
            return None
        value = STMChunks.stack_view(self.state)[3][:, :width]
        return self._snapshot_tensor(value, detach=detach)

    def snapshot_concept_rows(self, detach=False):
        width = min(int(self._max_depth_host), self.capacity)
        if width <= 0:
            return None
        value = STMChunks.stack_view(self.state)[4][:, :width]
        return self._snapshot_tensor(value, detach=detach)

    def snapshot_concept_activations(self, detach=False):
        width = min(int(self._max_depth_host), self.capacity)
        if width <= 0:
            return None
        value = STMChunks.stack_view(self.state)[5][:, :width]
        return self._snapshot_tensor(value, detach=detach)

    def size(self, b):
        return int(self._depth[b].item())

    def is_full(self, b):
        return self.size(b) >= self.capacity

    def is_empty(self, b):
        return self.size(b) == 0

    def clear(self, b=None, *, hard=True):
        gate = torch.ones(self._active_batch, dtype=torch.bool, device=self._buffer.device) if b is None else (
            torch.arange(self._active_batch, device=self._buffer.device) == int(b))
        state = (STMChunks.clear(self.state, gate[:, None].expand_as(self.state[STMChunks.STACK])) if hard
                 else STMChunks.soft_reset_rows(self.state, gate))
        self.set_state(state)
        from ClauseScope import ClauseScope
        self._clause_state = ClauseScope.reset(self.ensure_clause_state(), gate)

@dataclass(frozen=True)
class Name:
    token: tuple
    reading: int | None


@dataclass(frozen=True)
class Event:
    occurrence: tuple
    verb: int
    role: int
    when: int
    before: torch.Tensor | None = None
    basis: torch.Tensor | int | None = None
    taught: bool = False
    after: torch.Tensor | None = None
    modifiers: tuple = ()
    modal_taught: bool = False
    case_value: torch.Tensor | None = None
    verb_meaning: torch.Tensor | None = None


class Timeline(NamedTuple):
    events: tuple
    value: torch.Tensor


@dataclass(frozen=True)
class Token:
    occurrences: tuple
    kind: int | None
    count: int
    when: int
    timelines: tuple
    native: int | None = None
    pending: bool = False
    symbol: int | None = None
    transient: int | None = None
    where: torch.Tensor | None = None
    order: int | None = None
    support: dict = field(default_factory=dict)
    mass: float = 0.
    readings: tuple = ()
    refinement: dict = field(default_factory=dict)

    @property
    def value(self):
        return self.timelines[0].value

    @property
    def events(self):
        return self.timelines[0].events

    def with_actual(self, *, value=None, events=None, **metadata):
        actual = Timeline(self.events if events is None else events,
                          self.value if value is None else value)
        return replace(self, timelines=(actual, *self.timelines[1:]), **metadata)


class MediumTermMemory(MemoryStore):
    """Names share an identity read from its references until its first event.

    Mutation belongs to a kept close. The pool ends the sentence's graph;
    trial-local values live in the shared STM chunks. After an event, each
    timeline retains its state. Neither read changes the stored layout.
    Mention membership, used by the settled-join guard, excludes events.
    """

    def __init__(self, capacity=64, horizon=16, *, n_vectors=1,
                 field_width=None, match_cos=.8, mint_count=3):
        if capacity < 1:
            raise ValueError('mtmCapacity must be positive')
        if type(horizon) is not int or horizon < 1:
            raise ValueError('mtmHorizon must be positive')
        self.capacity = int(capacity)
        self.horizon = horizon
        self.closes, self.accesses = {}, {}
        self.entries = {}
        self.names = {}
        self.field_width = min(256, int(n_vectors)) if field_width is None else int(field_width)
        self.match_cos, self.mint_count = float(match_cos), int(mint_count)
        if self.field_width < 1 or not 0 < self.match_cos <= 1 or self.mint_count < 1:
            raise ValueError('invalid familiar-tier construction bounds')
        self.candidate_counter = 0
        # One reserved id for each possible unpromoted MTM token. These lie
        # beyond nVectors, never in the codebook, and are reused only after
        # their owner expires or acquires a durable symbol.
        self.transient_ids = torch.arange(int(n_vectors), int(n_vectors)+self.capacity, dtype=torch.long)
        self.transient_used = torch.zeros(self.capacity, dtype=torch.bool)

    def _allocate_id(self, requested=None):
        available = ~self.transient_used
        if requested is not None:
            available &= self.transient_ids == int(requested)
        seats = available.nonzero().flatten()
        if not len(seats):
            raise RuntimeError('identity transient id pool exhausted or requested id already in use')
        index = int(seats[0])
        self.transient_used[index] = True
        return int(self.transient_ids[index])

    def _release_id(self, entry):
        if entry.transient is not None:
            self.transient_used[self.transient_ids == entry.transient] = False

    @staticmethod
    def _where(where, previous=None):
        if where is None:
            return torch.zeros(4) if previous is None else MemoryStore._snapshot_tensor(previous)
        if not torch.is_tensor(where) or where.shape != (4,) or not torch.isfinite(where).all():
            raise ValueError('identity where requires four finite percept coordinates')
        return MemoryStore._snapshot_tensor(where)

    def promote(self, token, symbol):
        key = self.token_of(token)
        if key is None:
            raise ValueError('promotion requires a retained token')
        entry = self.entries[key]
        self._release_id(entry)
        self.entries[key] = replace(entry, symbol=int(symbol), transient=None, pending=False)
        reference = getattr(self, '_owner', None)
        owner = None if reference is None else reference()
        if owner is not None:
            owner._priming_edge_revision = None
            from FamiliarTier import class_members
            alloc = owner._concept_allocator
            affected = {parent for occurrence in entry.occurrences
                        for kind, parent in alloc.parts.wholes((OCCURRENCE, occurrence[-1]))
                        if kind == SYMBOL and parent in alloc.raised}
            for parent in affected:
                members = class_members(owner, parent)
                if members:
                    owner.cover_members(parent, members)

    @staticmethod
    def _value(value):
        if (not torch.is_tensor(value) or value.ndim != 1 or not value.is_floating_point()
                or not bool(torch.isfinite(value).all())):
            raise ValueError('an identity requires one finite working value')
        return MemoryStore._snapshot_tensor(value)

    def token_of(self, occurrence):
        if occurrence in self.entries:
            return occurrence
        name = self.names.get(occurrence)
        if name is not None:
            return name.token
        if isinstance(occurrence, int):
            candidates = [key for key, entry in self.entries.items() if occurrence in (entry.native, entry.symbol, entry.transient)]
            return candidates[0] if len(candidates) == 1 else None
        return next((key for key, entry in self.entries.items()
                     if any(event.occurrence == occurrence and event.role == 0
                            for timeline in entry.timelines for event in timeline.events)), None)

    def keep_access(self, reference, stream):
        """Mark an actual use at the kept close before another seat is opened."""
        key = self.token_of(reference)
        if key is not None:
            self.accesses.setdefault(key, {})[stream] = self.closes.get(stream, 0) + 1

    def _room(self, retain=()):
        if len(self.entries) >= self.capacity:
            protected = {self.token_of(reference) for reference in retain}
            def recency(key):
                uses = self.accesses.get(key, {})
                current = any(tick > self.closes.get(stream, 0) for stream, tick in uses.items())
                return current, max(uses.values(), default=-1), self.entries[key].when
            oldest = min((key for key in self.entries if key not in protected), key=recency, default=None)
            if oldest is None:
                raise RuntimeError(f'MTM capacity {self.capacity} exhausted by pending definition operands')
            self._detach(oldest)

    def _support(self, support):
        import math
        if any(type(row) is not int or not 0 <= row < int(self.transient_ids[0])
               or not math.isfinite(float(weight)) or float(weight) < 0 for row, weight in support.items()):
            raise ValueError('candidate support requires nonnegative finite inventory evidence')
        return dict(sorted(((row, float(weight)) for row, weight in support.items() if weight > 0),
                           key=lambda item: (-item[1], item[0]))[:self.field_width])

    def open_candidate(self, key=None, *, order, when, value, support):
        value, support = self._value(value), self._support(support)
        if type(order) is not int or order < 0 or not support:
            raise ValueError('a candidate requires a witnessed order and support')
        expected = ('candidate', self.candidate_counter)
        if key is not None and key != expected:
            raise ValueError('candidate key must use the pool counter')
        self._room()
        self.candidate_counter += 1
        self.entries[expected] = Token((), None, 1, int(when), (Timeline((), value),),
            transient=self._allocate_id(), pending=True, where=self._where(None), order=order,
            support=support, mass=sum(support.values()))
        return expected

    def match_candidate(self, order, value):
        value = self._value(value)
        scored = [(float(torch.nn.functional.cosine_similarity(self.value_of(key).to(value), value, dim=0)), key)
                  for key, entry in self.entries.items() if entry.order == order]
        score, key = max(scored, key=lambda item: item[0], default=(-1., None))
        return key if score >= self.match_cos else None

    def update_candidate(self, key, *, when, value, support):
        entry = self.entries[key]
        if entry.order is None:
            raise ValueError('candidate update requires a candidate')
        value, support = self._value(value), self._support(support)
        if value.shape != entry.value.shape or not support:
            raise ValueError('candidate update differs from its construction')
        mass = sum(support.values())
        merged = dict(entry.support)
        for row, weight in support.items():
            merged[row] = merged.get(row, 0.) + weight
        self.entries[key] = entry.with_actual(value=torch.lerp(entry.value.to(value), value,
            mass / (entry.mass + mass)), count=entry.count+1, when=int(when),
            mass=entry.mass+mass, support=self._support(merged))
        return key

    def read_candidates(self, owner):
        """A differentiable, sparse read through the current member codes."""
        result = {}
        for key, entry in self.entries.items():
            if entry.order is not None:
                rows = list(entry.support)
                codes = owner.similarity_codebook.lookup_rows(rows)
                weights = codes.new_tensor(list(entry.support.values()))
                result[key] = (codes * weights[:, None]).sum(0) / weights.sum()
        return result

    def record_reading(self, key, stream, pair):
        """One two-lane witness per kept-close clock, retained within the horizon."""
        entry = self.entries[key]
        clock = self.closes.get(stream, 0)+1
        import math
        pair = tuple(float(x) for x in pair)
        if len(pair) != 2 or any(not math.isfinite(x) or x < 0 for x in pair):
            raise ValueError('candidate reading requires two finite nonnegative lanes')
        # Several focal rows can identify one candidate in the same field.
        # Retain both lanes independently; the last focal must not erase one.
        prior = next((item[2:] for item in entry.readings if item[:2] == (stream, clock)), (0., 0.))
        pair = tuple(max(a, b) for a, b in zip(pair, prior))
        retained = tuple(item for item in entry.readings
            if item[:2] != (stream, clock) and
            self.closes.get(item[0], 0)+1-item[1] <= self.horizon)
        readings = (*retained, (stream, clock, *pair))
        self.entries[key] = replace(entry, readings=readings,
            count=len(readings) if entry.order is not None else entry.count)

    def promote_candidates(self, owner):
        from FamiliarTier import promote_candidates
        return promote_candidates(owner)

    def reading_of(self, occurrence):
        name = self.names.get(occurrence)
        return None if name is None else name.reading

    def value_of(self, reference, *, timeline=0, live_candidate=False, references=None):
        """Read held state after an event, otherwise join the live references.

        The model's meanings are detached at this host boundary, just as the
        stored MTM state is. Standalone numerical pools have no dictionary or
        occurrence owner and retain their explicit value API.
        """
        token = self.token_of(reference)
        if token is None:
            return None
        entry = self.entries[token]
        if type(timeline) is not int or not 0 <= timeline < len(entry.timelines):
            raise ValueError(f'identity {token!r} has no timeline {timeline}')
        stored = entry.timelines[timeline].value
        owner_reference = getattr(self, '_owner', None)
        if owner_reference is None:
            from Occurrence import AddressIndex
            held_names = [(address, name.reading) for address, name in self.names.items()
                          if name.token == token]
            semantic = any(reading is not None or AddressIndex.key(address) != address or
                           isinstance(address, tuple) and address[:1] in (('name',), ('stm',), ('sym',))
                           for address, reading in held_names)
            if semantic or entry.native is not None or entry.symbol is not None:
                raise ValueError(f'identity {token!r} holds references without an owner')
            return stored
        owner = owner_reference()
        if owner is None:
            raise ValueError(f'identity {token!r} has lost its reference owner')
        if timeline or any(branch.events for branch in entry.timelines):
            return stored
        derived = getattr(owner.similarity_codebook, 'mereology', None)
        start = 0 if derived is None else derived.percept_event_width
        if getattr(derived, 'meaning_pairs', None) == 0 or start == stored.numel():
            return stored
        if entry.order is not None:
            # Consolidation candidates have weighted concept support rather
            # than mention names. Their existing live read remains separate.
            from FamiliarTier import centroid
            value = centroid(owner, entry.support)
            return value if live_candidate else self._value(value)
        if references is None:
            references = [(address, name.reading) for address, name in self.names.items()
                          if name.token == token]
        held = tuple(dict.fromkeys(cid for cid in (entry.native, entry.symbol) if cid is not None))
        if not references and not held:
            raise ValueError(f'identity {token!r} has no name or mention references')
        store, joined = owner._closed_clause_store(), None
        alloc = owner._concept_allocator
        for address, reading in (*references, *((('sym', cid), cid) for cid in held)):
            is_word = reading is not None and 'word' in alloc.symbol_types.get(reading, ())
            if is_word or address[:1] == ('sym',):
                concept = owner.interpret.referent(reading) if is_word else reading
                row = owner._csw_row_of(concept)
                if row is None:
                    raise ValueError(f'identity {token!r} reference {address!r} has unavailable reading {reading}, concept {concept}')
                value = owner.similarity_codebook.lookup_rows([row])[0]
            else:
                location = None if store is None else store.location(address)
                if not isinstance(location, int):
                    raise ValueError(f'identity {token!r} has unavailable occurrence reference {address!r}')
                value = store.point_of_row(address[-1])
            if value is None:
                raise ValueError(f'identity {token!r} reference {address!r} has no content')
            if value.shape != stored.shape:
                raise ValueError(f'identity {token!r} reference {address!r} has a different content width')
            value = self._value(value).to(stored)
            joined = value if joined is None else torch.maximum(joined, value)
        current = self._snapshot_tensor(stored)
        current[start:] = joined[start:]
        return current

    def open(self, occurrence, *, kind, when, value, reading=None, native=None, pending=None, where=None,
             transient=None, retain=()):
        value = self._value(value)
        previous = self.reading_of(occurrence)
        if previous is not None:
            reading = previous
        pending = kind is None if pending is None else pending
        if type(pending) is not bool or (pending and kind not in (None, NULL_CONCEPT)):
            raise ValueError('a pending identity has no resolved kind other than the null concept')
        if self.token_of(occurrence) != occurrence:
            self._detach(occurrence)
        if occurrence in self.entries:
            entry = self.entries[occurrence]
            self.entries[occurrence] = entry.with_actual(count=entry.count + 1,
                when=when, value=torch.maximum(self.value_of(occurrence).to(value), value),
                pending=entry.pending and pending, where=self._where(where, entry.where))
            self.names[occurrence] = Name(occurrence, reading)
            return occurrence
        self._room(retain)
        identity = self._allocate_id(transient)
        self.entries[occurrence] = Token((occurrence,), kind, 1, when, (Timeline((), value),),
            native=native, pending=pending, transient=identity, where=self._where(where))
        self.names[occurrence] = Name(occurrence, reading)
        return occurrence

    def named(self, word, *, value, when=0):
        """Read a name's pending identity; first sight has no kept occurrence."""
        word = int(word)
        named = self.names.get(('name', word))
        if named is not None and named.token in self.entries:
            return named.token
        key = next((name.token for name in reversed(tuple(self.names.values()))
                    if name.reading == word and name.token in self.entries), None)
        if key is not None:
            self.names['name', word] = Name(key, word)
            return key
        key = ('name', word)
        self._room()
        self.entries[key] = Token((), None, 0, int(when),
            (Timeline((), self._value(value)),), pending=True,
            transient=self._allocate_id(), where=self._where(None))
        self.names[key] = Name(key, word)
        return key

    def bind(self, antecedent, occurrence, *, kind, when, value=None, reading=None, where=None):
        token = self.token_of(antecedent)
        if token is None:
            # Losing track does not resurrect an evicted introduction.
            return self.open(occurrence, kind=kind, when=when, value=value, reading=reading, where=where)
        # Binding can move a retained occurrence to a new token. Its
        # observed reading belongs to the occurrence, including after a move.
        previous = self.reading_of(occurrence)
        if previous is not None:
            reading = previous
        if self.token_of(occurrence) != token:
            self._detach(occurrence)
        entry = self.entries[token]
        uses = entry.occurrences
        if occurrence not in uses:
            uses += (occurrence,)
        current = self.value_of(token)
        joined = current if value is None else torch.maximum(current.to(value), self._value(value))
        self.entries[token] = entry.with_actual(occurrences=uses,
            kind=kind if entry.kind is None else entry.kind,
            pending=entry.pending and kind in (None, NULL_CONCEPT),
            count=entry.count + 1, when=when, value=joined, where=self._where(where, entry.where))
        self.names[occurrence] = Name(token, reading)
        return token

    def alternative(self, token, *, when, before=None):
        """Continue the latest branch, opening it from the actual prefix once."""
        key = self.token_of(token)
        if key is None:
            raise ValueError('an alternative requires a retained identity')
        entry = self.entries[key]
        if len(entry.timelines) == 1:
            value = self.value_of(key).clone() if before is None else self._value(before)
            if value.shape != entry.value.shape:
                raise ValueError('alternative value has a different width')
            self.entries[key] = replace(entry, when=when,
                timelines=(*entry.timelines, Timeline(entry.events, value)))
        return len(self.entries[key].timelines) - 1

    def update(self, token, value, *, when, timeline=0):
        key = self.token_of(token)
        if key is None:
            raise ValueError('cannot update an unretained identity')
        entry = self.entries[key]
        if type(timeline) is not int or not 0 <= timeline < len(entry.timelines):
            raise ValueError('cannot update an absent timeline')
        if not any(branch.events for branch in entry.timelines):
            raise ValueError(f'cannot update identity {key!r} without an event; retain its written reference')
        value = self._value(value)
        if value.shape != entry.value.shape:
            raise ValueError('timeline value has a different width')
        self.entries[key] = replace(entry, when=when, timelines=tuple(
            Timeline(item.events, value) if index == timeline else item
            for index, item in enumerate(entry.timelines)))

    def event(self, token, occurrence, *, verb, role, when, before=None, basis=None,
              after=None, modifiers=(), timeline=0, case_value=None, verb_meaning=None):
        key = self.token_of(token)
        if key is None or role not in (0, 2) or type(verb) is not int:
            raise ValueError('an event requires a retained identity, verb and participant role')
        entry = self.entries[key]
        if type(timeline) is not int or not 0 <= timeline < len(entry.timelines):
            raise ValueError('an event requires a retained timeline')
        old = entry.timelines[timeline]
        current = (self.value_of(key), *(item.value for item in entry.timelines[1:]))
        if case_value is not None and (self._value(case_value).shape != old.value.shape or
                                     bool(((case_value < 0) | (case_value > 1)).any())):
            raise ValueError('an event case test requires its native content shape and chart')
        event = Event(occurrence, verb, role, when,
            None if before is None else self._value(before),
            self._value(basis) if torch.is_tensor(basis) else basis,
            after=None if after is None else self._value(after),
            modifiers=tuple(self._value(value) for value in modifiers),
            case_value=None if case_value is None else self._value(case_value),
            verb_meaning=None if verb_meaning is None else self._value(verb_meaning))
        previous = next((item for item in old.events
                         if (item.occurrence, item.role) == (occurrence, role)), None)
        if previous is not None:
            event = replace(event, taught=previous.taught, modal_taught=previous.modal_taught)
            events = tuple(event if item is previous else item for item in old.events)
        else:
            events = (*old.events, event)
        self.entries[key] = replace(entry, when=when, timelines=tuple(
            Timeline(events if index == timeline else item.events, current[index])
            for index, item in enumerate(entry.timelines)))

    def _detach(self, occurrence):
        """Detach a mention; an event with this address has its own owner."""
        if occurrence in self.entries and self.entries[occurrence].order is not None:
            self._release_id(self.entries.pop(occurrence))
            self.accesses.pop(occurrence, None)
            return
        name = self.names.pop(occurrence, None)
        if name is not None:
            token, entry = name.token, self.entries[name.token]
            if token == occurrence:
                self._release_id(entry)
                del self.entries[token]
                self.accesses.pop(token, None)
                self.names = {address: name for address, name in self.names.items() if name.token != token}
            else:
                self.entries[token] = replace(entry, occurrences=tuple(
                    address for address in entry.occurrences if address != occurrence))
    def forget(self, occurrence):
        self._detach(occurrence)
        # One event can witness a subject and an object in different identities.
        for token, entry in tuple(self.entries.items()):
            self.entries[token] = replace(entry, timelines=tuple(
                Timeline(tuple(event for event in timeline.events if event.occurrence != occurrence), timeline.value)
                for timeline in entry.timelines))

    def merge(self, token, into, *, when):
        """Join a pending object without moving the antecedent's existing uses."""
        source, target = self.token_of(token), self.token_of(into)
        if source is None or target is None:
            raise ValueError('MTM merge requires two retained tokens')
        if source == target:
            return target
        previous, antecedent = self.entries[source], self.entries[target]
        self.entries[target] = antecedent.with_actual(
            occurrences=antecedent.occurrences + previous.occurrences,
            count=antecedent.count + previous.count, when=when,
            kind=previous.kind if antecedent.kind is None else antecedent.kind,
            pending=antecedent.pending and previous.pending,
            value=torch.maximum(self.value_of(target), self.value_of(source).to(antecedent.value)),
            events=antecedent.events + tuple(event for event in previous.events
                if not any((other.occurrence, other.role) == (event.occurrence, event.role)
                           for other in antecedent.events)))
        self.entries[target] = replace(self.entries[target], timelines=(
            *self.entries[target].timelines, *previous.timelines[1:]))
        for occurrence, name in tuple(self.names.items()):
            if name.token == source:
                self.names[occurrence] = replace(name, token=target)
        refinement = dict(self.entries[target].refinement)
        refinement['mentions'] = {**previous.refinement.get('mentions', {}),
                                  **refinement.get('mentions', {})}
        self.entries[target] = replace(self.entries[target], refinement=refinement)
        for stream, clock in self.accesses.pop(source, {}).items():
            retained = self.accesses.setdefault(target, {})
            retained[stream] = max(clock, retained.get(stream, 0))
        self._release_id(previous)
        del self.entries[source]
        return target

    def close(self, store, *, stream=None, references=()):
        """Advance once after a kept sentence; staging and reads never call this."""
        clock = self.closes.get(stream, 0) + 1
        self.closes[stream] = clock
        from DocumentCover import latest, segments
        current = None if stream is None or store is None else latest(store, stream)
        membership = {} if store is None else segments(store, stream)
        for key in self.entries:
            if self.entries[key].order is not None:
                continue  # A candidate belongs only to streams that read it.
            row = None if store is None else store.location(key)
            if current is not None and current not in membership.get(row, ()):
                continue
            # Uncovered legacy/fixture rows retain the original stream aging.
            if (stream is not None and isinstance(row, int) and bool(store.stream[row].any())
                    and (stream >= store.stream.shape[1] or not bool(store.stream[row, stream]))):
                continue
            self.accesses.setdefault(key, {stream: clock})
        for reference in references:
            key = self.token_of(reference)
            if key is not None:
                self.accesses.setdefault(key, {})[stream] = clock
                entry = self.entries[key]
                if entry.order is None and reference in self.names:
                    refinement = dict(entry.refinement)
                    mentions = {address: seen for address, seen in refinement.get('mentions', {}).items()
                                if self.closes.get(seen[0], 0)-seen[1] <= self.horizon}
                    mentions.setdefault(reference, (stream, clock))
                    refinement['mentions'] = mentions
                    self.entries[key] = replace(entry, refinement=refinement)
        for key, accesses in tuple(self.accesses.items()):
            if stream in accesses and clock - accesses[stream] > self.horizon:
                del accesses[stream]
                if not accesses:
                    self._detach(key)
        for key, entry in tuple(self.entries.items()):
            readings = tuple(item for item in entry.readings
                if self.closes.get(item[0], 0)-item[1] <= self.horizon)
            if readings != entry.readings:
                self.entries[key] = replace(entry, readings=readings,
                    count=len(readings) if entry.order is not None else entry.count)

    def end_document(self, cover, *, store):
        """A document ends its cover, not the epoch's familiarity horizon."""
        return None

    def state(self):
        """Checkpoint only committed values; export owns every tensor copy."""
        def snapshot(value):
            return self._snapshot_tensor(value, cpu=True) if torch.is_tensor(value) else value
        def event_state(event):
            return (event.occurrence, event.verb, event.role, event.when,
                snapshot(event.before), snapshot(event.basis), event.taught,
                snapshot(event.after), tuple(map(snapshot, event.modifiers)), event.modal_taught,
                snapshot(event.case_value), snapshot(event.verb_meaning))
        return dict(version=11, candidate_counter=self.candidate_counter, field_width=self.field_width,
            transient_ids=snapshot(self.transient_ids), transient_used=snapshot(self.transient_used),
            closes=tuple(self.closes.items()),
            accesses=tuple((key, tuple(value.items())) for key, value in self.accesses.items()),
            entries=tuple(dict(key=key, occurrences=entry.occurrences,
            kind=entry.kind, count=entry.count, when=entry.when,
            native=entry.native, pending=entry.pending, symbol=entry.symbol, transient=entry.transient, where=snapshot(entry.where),
            order=entry.order, support=dict(entry.support), mass=entry.mass,
            readings=entry.readings, refinement=copy.deepcopy(entry.refinement),
            timelines=tuple((tuple(map(event_state, timeline.events)), snapshot(timeline.value))
                            for timeline in entry.timelines)) for key, entry in self.entries.items()),
            names=tuple((mention, name.token, name.reading) for mention, name in self.names.items()))

    def restore(self, state, *, device=None):
        """Validate a complete pool before replacing its current context."""
        if state is None:
            self.entries, self.names = {}, {}
            self.closes, self.accesses = {}, {}
            self.transient_used.zero_()
            self.candidate_counter = 0
            return
        if not isinstance(state, dict) or state.get('version') not in (8, 9, 10, 11):
            raise ValueError('MTM checkpoint requires transient pool version 8, 9, 10 or 11')
        counter = state.get('candidate_counter', 0)
        if type(counter) is not int or counter < 0 or state.get('field_width', self.field_width) != self.field_width:
            raise ValueError('candidate checkpoint disagrees with construction')
        ids, used = state.get('transient_ids'), state.get('transient_used')
        if (not torch.is_tensor(ids) or ids.dtype != torch.long or not torch.equal(ids.cpu(), self.transient_ids)
                or not torch.is_tensor(used) or used.dtype != torch.bool or used.shape != self.transient_used.shape):
            raise ValueError('identity transient pool disagrees with construction')
        if state['version'] >= 6 and 'documents' in state:
            raise ValueError('identity checkpoint contains retired document provenance')
        def address(value):
            return (isinstance(value, tuple) and len(value) == 3 and value[0] == 'ltm'
                    and isinstance(value[1], str) and type(value[2]) is int)
        def named(value):
            return (state['version'] >= 10 and isinstance(value, tuple) and len(value) == 2
                    and value[0] == 'name' and type(value[1]) is int and value[1] > 0)
        def symbol(value):
            return value is None or type(value) is int and value > 0
        def sentence(value):
            return address(value) or (isinstance(value, tuple) and len(value) == 2
                and value[0] == 'sym' and type(value[1]) is int and value[1] > 0)
        if not all(isinstance(state.get(field), (tuple, list)) for field in ('entries', 'names')):
            raise ValueError('identity checkpoint requires entries and names')
        entries, names = {}, {}
        for record in state['entries']:
            required = ('timelines',) if state['version'] >= 5 else ('value', 'events')
            if not isinstance(record, dict) or not set(('key', 'occurrences', 'kind', 'count',
                    'when', 'native', *required)) <= record.keys():
                raise ValueError('incomplete identity checkpoint entry')
            key, uses = record['key'], tuple(record['occurrences'])
            candidate = (isinstance(key, tuple) and len(key) == 2 and key[0] == 'candidate'
                         and type(key[1]) is int and 0 <= key[1] < counter)
            if (key in entries or (candidate and uses) or (not candidate and
                    (not (address(key) or named(key)) or not all(map(address, uses))
                     or (not named(key) and (not uses or uses[0] != key))
                     or len(set(uses)) != len(uses)))):
                raise ValueError('invalid identity checkpoint membership')
            order, support = record.get('order'), self._support(record.get('support', {}))
            if (candidate and (type(order) is not int or order < 0 or not support)
                    or not candidate and (order is not None or support)
                    or len(support) != len(record.get('support', {}))):
                raise ValueError('invalid candidate checkpoint support')
            if (not symbol(record['kind']) or not symbol(record['native']) or
                    type(record['count']) is not int or record['count'] < len(uses) or
                    type(record['when']) is not int):
                raise ValueError('invalid identity checkpoint metadata')
            pending = record['kind'] is None if state['version'] == 1 else record.get('pending')
            if type(pending) is not bool or (pending and record['kind'] not in (None, NULL_CONCEPT)):
                raise ValueError('invalid identity checkpoint pending state')
            saved = record['timelines'] if state['version'] >= 5 else ((record['events'], record['value']),)
            if not isinstance(saved, (tuple, list)) or not saved:
                raise ValueError('identity checkpoint requires an actual timeline')
            timelines = []
            for timeline in saved:
                if not isinstance(timeline, (tuple, list)) or len(timeline) != 2:
                    raise ValueError('invalid identity checkpoint timeline')
                history, value = timeline
                value = self._value(value)
                expected = (timelines[0].value if timelines else
                            next(iter(entries.values())).value if entries else value)
                if value.shape != expected.shape:
                    raise ValueError('identity checkpoint values have different widths')
                if device is not None:
                    value = value.to(device)
                size = 12 if state['version'] >= 11 else 11 if state['version'] >= 7 else 10 if state['version'] >= 5 else 7 if state['version'] == 4 else 4
                if not isinstance(history, (tuple, list)) or any(
                        not isinstance(event, (tuple, list)) or len(event) != size for event in history):
                    raise ValueError('invalid identity checkpoint event')
                events = tuple(Event(*event) for event in history)
                for event in events:
                    if event.case_value is not None and bool(((event.case_value < 0) | (event.case_value > 1)).any()):
                        raise ValueError('invalid identity event case test')
                    if (type(event.taught) is not bool or type(event.modal_taught) is not bool
                            or not isinstance(event.modifiers, (tuple, list))
                            or (event.basis is not None and not torch.is_tensor(event.basis)
                                and (type(event.basis) is not int or event.basis < -1))):
                        raise ValueError('invalid identity event basis or teacher metadata')
                    for snapshot in (event.before, event.after, event.case_value, event.verb_meaning, *event.modifiers):
                        if snapshot is not None and self._value(snapshot).shape != value.shape:
                            raise ValueError('invalid identity event snapshot')
                    if torch.is_tensor(event.basis) and self._value(event.basis).shape != value.shape:
                        raise ValueError('invalid identity event snapshot')
                events = tuple(replace(event,
                    before=None if event.before is None else self._value(event.before).to(value),
                    after=None if event.after is None else self._value(event.after).to(value),
                    case_value=None if event.case_value is None else self._value(event.case_value).to(value),
                    verb_meaning=None if event.verb_meaning is None else self._value(event.verb_meaning).to(value),
                    modifiers=tuple(self._value(item).to(value) for item in event.modifiers),
                    basis=self._value(event.basis).to(value) if torch.is_tensor(event.basis) else event.basis)
                    for event in events)
                if (any(not sentence(event.occurrence) or event.role not in (0, 2) or
                        type(event.verb) is not int or event.verb <= 0 or type(event.when) is not int
                        for event in events) or len({(event.occurrence, event.role) for event in events}) != len(events)):
                    raise ValueError('invalid identity checkpoint event')
                timelines.append(Timeline(events, value))
            entries[key] = Token(uses, record['kind'], record['count'], record['when'],
                                 tuple(timelines), record['native'], pending, record.get('symbol'),
                                 record.get('transient'), self._where(record.get('where')),
                                 order, support, float(record.get('mass', 0.)),
                                 tuple(record.get('readings', ())), copy.deepcopy(record.get('refinement', {})))
            if not symbol(record.get('symbol')):
                raise ValueError('invalid identity checkpoint promoted symbol')
        for mention, token, reading in state['names']:
            if (not (address(mention) or named(mention)) or not (address(token) or named(token))
                    or not symbol(reading) or mention in names or token not in entries
                    or (mention not in entries[token].occurrences and not (named(mention) and reading == mention[1]))):
                raise ValueError('invalid identity checkpoint name')
            names[mention] = Name(token, reading)
        if (len(entries) > self.capacity or len(names) != sum(len(entry.occurrences)
                for entry in entries.values()) + sum(named(mention) for mention in names)):
            raise ValueError('identity checkpoint exceeds capacity or omits a name')
        clocks, accesses = {}, {}
        if state['version'] >= 3:
            try:
                clocks = dict(state['closes'])
                accesses = {key: dict(value) for key, value in state['accesses']}
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError('invalid identity checkpoint expiry metadata') from error
            if (any((stream is not None and (type(stream) is not int or stream < 0)) or
                    type(clock) is not int or clock < 0 for stream, clock in clocks.items())
                    or any(key not in entries or not value or any(stream not in clocks or
                        type(clock) is not int or not 0 <= clock <= clocks[stream]
                        for stream, clock in value.items()) for key, value in accesses.items())):
                raise ValueError('invalid identity checkpoint expiry metadata')
        claimed = [entry.transient for entry in entries.values() if entry.symbol is None]
        if (any(type(identity) is not int or identity not in self.transient_ids.tolist() for identity in claimed)
                or len(claimed) != len(set(claimed))
                or any(entry.transient is not None for entry in entries.values() if entry.symbol is not None)
                or not torch.equal(used.cpu(), torch.tensor([int(identity) in claimed for identity in self.transient_ids]))):
            raise ValueError('identity checkpoint transient assignments are inconsistent')
        self.transient_used.copy_(used.cpu())
        self.entries, self.names = entries, names
        self.closes, self.accesses = clocks, accesses
        self.candidate_counter = counter


class LTM0(OccurrenceRows, ClauseRows, EntryRead, MemoryStore, Layer):
    """Durable ideas: one paired event vector and independent evidence.

    LTM1 owns semantic chains. No role table or composition derivation is
    retained here. Semantic bindings preserve identity, scope and provenance;
    the percept band owns the sentence's model-time stamp.
    """

    ORIGIN_CONVERSATION = 0
    ORIGIN_PROVISIONED = 1
    ORIGIN_USER = 2

    KINDS = ("unverified", "fact", "question", "estimate", "observation", "inference", "cover")

    def __init__(self, nDim: int, capacity: int = 1024, content_width=None, *, percept_width=None,
                 stream_capacity=None, recurrence_capacity=None, relation_capacity=8192):
        super().__init__(nDim, nDim)
        self.nDim = int(nDim)
        self.capacity = int(capacity)
        # Cover initialization has its own checkpointed stream. Creating
        # a document must not change the chooser's global random draws.
        self._cover_generator = torch.Generator(device='cpu').manual_seed(0)
        self.register_buffer('cover_rng_state', self._cover_generator.get_state())
        self._cover_symbols = {}  # document digest -> persistent conceptual cover
        # Content ('what') width a reasoning reader slices to (the leading
        # band of the event vector); defaults to the full width. Stored for
        # forward-compat -- this class keeps the FULL idea vectors.
        self.content_width = int(content_width) if content_width else self.nDim
        # One idea per declared seat; the existing percept ends in four time coordinates.
        self.percept_width = int(percept_width if percept_width is not None else nDim)
        if not 0 <= self.percept_width <= self.nDim or self.capacity <= 0:
            raise ValueError('invalid LTM0 construction layout')
        self.when_start = self.percept_width - 4 if self.percept_width >= 8 else None
        self.register_buffer('percept_layout', torch.tensor(self.percept_width, dtype=torch.long))
        self.register_buffer('slots', torch.zeros(capacity, nDim))
        self.ltm_window = int(TheXMLConfig.get('architecture.ltmWindow', default=1024))
        if self.ltm_window <= 0:
            raise ValueError('ltmWindow must be a positive integer')
        # Ephemeral cued-read scratch: fixed at construction, never checkpointed.
        self.register_buffer('_cue_roles', torch.zeros(self.ltm_window, 3, nDim), persistent=False)
        self.register_buffer('_cue_role_mask', torch.zeros(self.ltm_window, 3, dtype=torch.bool), persistent=False)
        self.register_buffer('_cue_available', torch.zeros(self.ltm_window, dtype=torch.bool), persistent=False)

        self.addressing = nn.Module()
        # Source authority belongs to an input document, including during a
        # tensor-only restore before semantic sidecars have been attached.
        self.addressing.register_buffer('source_documents', torch.zeros(capacity, 32, dtype=torch.uint8))
        self.addressing.register_buffer('source_authorities', torch.zeros(capacity, dtype=torch.int8))
        # Production supplies ConceptualSpace.recurrenceCapacity, default nVectors:
        # one fixed summary seat per allowed reading, independent of episode count.
        # Standalone stores without a concept owner use their declared row bound.
        self.recurrence_capacity = capacity if recurrence_capacity is None else int(recurrence_capacity)
        if self.recurrence_capacity <= 0:
            raise ValueError('recurrenceCapacity must be positive')
        reading_cap = self.recurrence_capacity
        self.register_buffer('recurrence_version', torch.tensor(2, dtype=torch.long))
        self.addressing.register_buffer('recurrence_codes', torch.zeros(reading_cap, nDim))
        self.addressing.register_buffer('recurrence_documents', torch.zeros(reading_cap, dtype=torch.long))
        self.addressing.register_buffer('recurrence_weights', torch.zeros(reading_cap))
        self.addressing.register_buffer('recurrence_used', torch.zeros(reading_cap, dtype=torch.bool))
        self.addressing.register_buffer('recurrence_symbols', torch.full((reading_cap,), -1, dtype=torch.long))
        # One assignment per episode row; compaction moves these with the ideas.
        self.addressing.register_buffer('recurrence_readings', torch.full((capacity,), -1, dtype=torch.long))
        self._recurrence_identities = [None] * reading_cap
        # A batch must fit at least one close per stream in the declared
        # store. This bounds stream ids by capacity unless configured tighter.
        stream_capacity = capacity if stream_capacity is None else int(stream_capacity)
        if stream_capacity <= 0:
            raise ValueError('LTM0 stream capacity must be positive')
        self.addressing.register_buffer('stream', torch.zeros(capacity, int(stream_capacity), dtype=torch.bool))
        self.register_buffer('truth_plus', torch.zeros(capacity))
        self.register_buffer('truth_minus', torch.zeros(capacity))
        for name in ('forget_passes', 'forget_sentences', 'forget_last_sentence'):
            self.register_buffer(name, torch.tensor(0, dtype=torch.long))
        self.register_buffer('forget_cutoff', torch.tensor(0.))
        self.register_buffer('forget_pending', torch.tensor(False))
        # At most one observation id per LTM0 seat, rebuilt only after a write.
        self.register_buffer('_observation_ids', torch.full((capacity,), -1, dtype=torch.long), persistent=False)
        self._observation_revision = None
        self._observation_count = 0
        self.register_buffer('model_clock', torch.tensor(0, dtype=torch.long))
        self.register_buffer('truth_schema', torch.tensor(14, dtype=torch.long))
        self.register_buffer('surprise', torch.full((capacity,), -1.))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        # Per-row writer provenance (ORIGIN_*); default 0 = conversation so
        # the observe-site push needs no change.
        # Source text is checkpointed in semantic_extras. A bare state_dict
        # restore has no text until that sidecar is supplied.
        self._texts = []

        # Semantic provenance is separate from origin and signed evidence.
        # Scope/reference metadata rides the model's versioned sidecar; the
        # flag prevents a tensor-only restore from forgetting required scope.
        self.addressing.register_buffer('address_keys', torch.full((capacity,), -1, dtype=torch.long))
        self.addressing.register_buffer('document_keys', torch.zeros(capacity, 32, dtype=torch.uint8))
        self.addressing.register_buffer('sentence_content_keys', torch.zeros(capacity, 32, dtype=torch.uint8))
        self.addressing.register_buffer('sentence_index', torch.full((capacity,), -1, dtype=torch.long))
        self.register_buffer('access_count', torch.zeros(capacity, dtype=torch.long))
        self.addressing.register_buffer('metadata_required', torch.zeros(capacity, dtype=torch.bool))
        self.addressing.register_buffer('semantic_fingerprint', torch.zeros(capacity, 32, dtype=torch.uint8))
        from Occurrence import document_digest
        self._direct_kinds = {document_digest(('direct', kind)): kind for kind in self.KINDS}
        self._document_sources = {}
        self._semantic_rows = {}
        # Estimate/observation provenance has the same durable owner as the
        # rows themselves. It is sidecar data, not a second LTM: occurrence
        # IDs, role vectors and evidence kind remain in the buffers above.
        self._expectation_rows = {}
        self._index_occurrences = AddressIndex()
        self.register_load_state_dict_post_hook(self._restore_address_index)
        object.__setattr__(self, 'parts', LTM1(relation_capacity))
        import weakref
        self.parts._occurrence_store = weakref.ref(self)

    @staticmethod
    def _restore_address_index(store, _incompatible):
        store.rebuild_addresses()
        keys = store.addressing.source_documents
        grades = store.addressing.source_authorities
        occupied = grades.nonzero().flatten()
        recorded = {bytes(key).hex(): grade for key, grade in
                    zip(keys[occupied].tolist(), grades[occupied].tolist())}
        live = {bytes(key).hex() for key in store.document_keys[:len(store)].tolist()}
        if (len(recorded) != len(occupied) or not recorded.keys() <= live or
                any(grade not in (store.ORIGIN_PROVISIONED, store.ORIGIN_USER) for grade in recorded.values())):
            raise ValueError('invalid source document in LTM0 addressing checkpoint')
        store._document_sources = recorded

    def _replace_document_sources(self, recorded):
        if recorded == self._document_sources:
            return
        if len(recorded) > self.capacity:
            raise OverflowError('source document count exceeds LTM0 construction capacity')
        keys, grades = self.addressing.source_documents, self.addressing.source_authorities
        keys.zero_()
        grades.zero_()
        if recorded:
            keys[:len(recorded)] = keys.new_tensor([list(bytes.fromhex(key)) for key in recorded])
            grades[:len(recorded)] = grades.new_tensor(list(recorded.values()))
        self._document_sources = recorded

    @property
    def when(self):
        return self.slots[:, :0] if self.when_start is None else self.slots[:, self.when_start:self.when_start + 4]

    @property
    def address_keys(self):
        return self.addressing.address_keys

    @property
    def document_keys(self):
        return self.addressing.document_keys

    @property
    def sentence_content_keys(self):
        return self.addressing.sentence_content_keys

    @property
    def sentence_index(self):
        return self.addressing.sentence_index

    @property
    def stream(self):
        return self.addressing.stream

    @property
    def metadata_required(self):
        return self.addressing.metadata_required

    @property
    def semantic_fingerprint(self):
        return self.addressing.semantic_fingerprint

    @property
    def recurrence_readings(self):
        return self.addressing.recurrence_readings

    @property
    def recurrence_codes(self):
        return self.addressing.recurrence_codes

    @property
    def recurrence_documents(self):
        return self.addressing.recurrence_documents

    @property
    def recurrence_weights(self):
        return self.addressing.recurrence_weights

    @property
    def recurrence_used(self):
        return self.addressing.recurrence_used

    @property
    def recurrence_symbols(self):
        return self.addressing.recurrence_symbols

    def clear_recurrence(self, *, keep_learned=False):
        """Clear episode evidence. Only an admitted reading remembers its symbol."""
        for name in ('recurrence_documents', 'recurrence_weights', 'recurrence_used'):
            getattr(self, name).zero_()
        if keep_learned:
            self.recurrence_codes.masked_fill_(self.recurrence_symbols.lt(0)[:, None], 0.)
        else:
            self.recurrence_codes.zero_()
            self.recurrence_symbols.fill_(-1)
        self.recurrence_readings.fill_(-1)
        self._recurrence_identities[:] = [None] * self.recurrence_capacity

    def source_document(self, row):
        """Authority belongs to the input document. It is not an occurrence property."""
        key = bytes(self.document_keys[int(row)].tolist()).hex()
        return self._document_sources.get(key, self.ORIGIN_CONVERSATION)

    def order_of_row(self, row, seen=None):
        """A native occurrence is concrete unless it carries a higher-order referent."""
        row = int(row)
        address = int(self.address_keys[row])
        seen = set() if seen is None else set(seen)
        if address in seen:
            return -1
        seen.add(address)
        bindings = dict(self._semantic_rows.get(address, {}).get('bindings', ()))
        ref = bindings.get('_identity')
        if not ref:
            return 0
        if ref[0] == 'ltm':
            child = self.location(ref)
            return -1 if child is None else self.order_of_row(child, seen)
        if ref[0] == 'sym':
            owner = self.__dict__.get('_concept_owner')
            if owner is not None and owner() is not None:
                return owner()._concept_source_order(ref[1], seen)
        return 0

    def kind_of(self, row):
        i = int(row)
        address = int(self.address_keys[i])
        provenance = self._expectation_rows.get(address, {})
        if provenance.get('kind') == 'estimate':
            return 'estimate'
        bindings = dict(self._semantic_rows.get(address, {}).get('bindings', ()))
        identity = bindings.get('_identity')
        if identity and identity[0]=='sym' and identity[1] in self._cover_symbols.values():
            return 'cover'
        if bindings.get('_open_references'):
            from ThoughtReferences import has_open_slots
            if has_open_slots(self.raw_meaning_of(i)):
                return 'question'
        # Direct typed input documents have stable addresses. In a native
        # document, an affirmed occurrence is an observation unless its
        # document supplied independent fact authority.
        document = bytes(self.document_keys[i].tolist())
        if document in self._direct_kinds:
            direct = self._direct_kinds[document]
            if (direct == 'fact' and self.source_document(i) != self.ORIGIN_CONVERSATION
                    and not (bool(self.truth_plus[i]) or bool(self.truth_minus[i]))):
                # Withdrawing an external source removes its authority. A
                # direct fact document can still describe an unknown (0, 0).
                return 'unverified'
            return direct
        if bool(self.truth_plus[i] > 0) or bool(self.truth_minus[i] > 0):
            if bindings.get('_producing_operation') or bindings.get('_thought_witnesses'):
                return 'inference'
            return 'fact' if self.source_document(i) != self.ORIGIN_CONVERSATION else 'observation'
        return 'unverified'

    def kind_codes(self, rows):
        return self.count.new_tensor([self.KINDS.index(self.kind_of(i)) for i in rows.tolist()])

    def polarity_of(self, row):
        ref = dict(self._semantic_rows.get(int(self.address_keys[row]), {}).get('bindings', ())).get('_identity')
        if ref and ref[0] == 'sym':
            owner = self.__dict__.get('_concept_owner')
            if owner is not None and owner() is not None and 'negative' in owner()._concept_allocator.symbol_types.get(ref[1], ()):
                return False
        return True

    def observation_ids(self):
        revision = (self.address_keys._version, self.truth_plus._version, self.truth_minus._version, len(self))
        if revision != self._observation_revision:
            rows = [i for i in range(len(self)) if self.kind_of(i) in ('observation', 'fact')]
            self._observation_count = len(rows)
            self._observation_ids[:len(rows)].copy_(self.address_keys[rows])
            self._observation_revision = revision
        return self._observation_ids[:self._observation_count]

    @staticmethod
    def _context_fingerprint(context, text, expectation=None):
        """Bind sidecar content to its tensor-owned occurrence without duplicating it."""
        context = context or {}
        content = {"bindings": context.get("bindings", ()),
                   "scope": context.get("scope", ()), "text": text}
        # Legacy records keep their old fingerprint exactly. Only rows that
        # actually carry expectation provenance add this key.
        if expectation is not None:
            content["expectation"] = expectation
        return LTM0._fingerprint_payload(content)

    @staticmethod
    def _fingerprint_payload(content):
        """Hash metadata, including a legacy payload while validating migration."""
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
        return list(hashlib.sha256(encoded).digest())

    @staticmethod
    def _freeze_expectation_value(value):
        """Make stream/document provenance immutable and checkpoint-safe."""
        if isinstance(value, dict):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("expectation provenance mapping keys must be strings")
            return tuple((key, LTM0._freeze_expectation_value(item))
                         for key, item in sorted(value.items()))
        if isinstance(value, (tuple, list)):
            return tuple(LTM0._freeze_expectation_value(item)
                         for item in value)
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise TypeError("expectation provenance must contain finite scalar values or typed references")

    @staticmethod
    def _validated_expectation_reference(value, namespace, *, optional=False):
        if value is None and optional:
            return None
        if (isinstance(value, tuple) and len(value) == 2 and value[0] == 'sym'
                and type(value[1]) is int and value[1] > 0):
            return value
        if (not isinstance(value, tuple) or len(value) != 3
                or value[0] != "ltm" or value[1] != namespace
                or type(value[2]) is not int or value[2] in (-1, 0)):
            raise ValueError("expectation provenance requires a local LTM occurrence reference")
        return value

    @classmethod
    def _normalise_expectation_provenance(cls, value, namespace):
        """Validate one explicit estimate/observation relationship record."""
        if value is None:
            return None
        if not isinstance(value, dict) or not isinstance(value.get("kind"), str):
            raise ValueError("invalid expectation provenance")
        kind = value["kind"]
        if kind == "estimate":
            expected = {"kind", "source_occurrences", "stream", "document",
                        "intended_occurrence", "presence_logits"}
            if (set(value) != expected
                    or not isinstance(value["source_occurrences"], tuple)):
                raise ValueError("invalid estimate provenance")
            sources = tuple(cls._validated_expectation_reference(source, namespace)
                            for source in value["source_occurrences"])
            intended = cls._validated_expectation_reference(
                value["intended_occurrence"], namespace, optional=True)
            logits = value["presence_logits"]
            if (not isinstance(logits, tuple) or len(logits) != 3
                    or any(not isinstance(item, (int, float))
                           or not math.isfinite(float(item)) for item in logits)):
                raise ValueError("estimate provenance requires three finite presence logits")
            return {
                "kind": "estimate",
                "source_occurrences": sources,
                "stream": cls._freeze_expectation_value(value["stream"]),
                "document": cls._freeze_expectation_value(value["document"]),
                "intended_occurrence": intended,
                "presence_logits": tuple(float(item) for item in logits),
            }
        if kind == "observation":
            if set(value) - {'previous_estimates'} != {"kind", "estimate_occurrence"}:
                raise ValueError("invalid observation expectation provenance")
            return {
                "kind": "observation",
                "estimate_occurrence": cls._validated_expectation_reference(
                    value["estimate_occurrence"], namespace),
                **({'previous_estimates': tuple(cls._validated_expectation_reference(ref, namespace)
                    for ref in value['previous_estimates'])} if 'previous_estimates' in value else {}),
            }
        raise ValueError("unknown expectation provenance kind")

    @staticmethod
    def _expectation_references(provenance):
        """All durable occurrence edges carried by one expectation record."""
        if provenance is None:
            return ()
        if provenance["kind"] == "estimate":
            return tuple(provenance["source_occurrences"]) + tuple(
                reference for reference in (provenance["intended_occurrence"],)
                if reference is not None)
        return (*provenance.get('previous_estimates', ()), provenance["estimate_occurrence"])

    def _validate_expectation_links(self, records=None):
        """Verify reference availability and bidirectional pair integrity."""
        records = self._expectation_rows if records is None else records
        namespace = self.address_domain
        identifiers = set(int(identifier) for identifier in self.address_keys[:len(self)].tolist())
        for identifier, provenance in records.items():
            if identifier not in identifiers:
                raise ValueError("expectation provenance names an unavailable occurrence")
            for reference in self._expectation_references(provenance):
                self._validated_expectation_reference(reference, namespace)
                if (not self.parts.contains(reference[1]) if reference[0] == 'sym'
                        else reference[2] not in identifiers):
                    raise ValueError("expectation provenance occurrence is unavailable")
            if provenance["kind"] != "estimate":
                continue
            target = provenance["intended_occurrence"]
            if target is None:
                continue
            if target[0] == 'sym':
                # LTM1 has only its six columns. Its inverse expectation
                # link is a query over these retained estimate records.
                continue
            inverse = records.get(target[2])
            if (inverse is None or inverse["kind"] != "observation"
                    or identifier not in {ref[2] for ref in self._expectation_references(inverse)}):
                raise ValueError("expectation provenance pair is not bidirectionally linked")

    def _update_semantic_fingerprint(self, idx):
        """Refresh the one integrity binding for context, text and prediction links."""
        i = int(idx)
        identifier = int(self.address_keys[i])
        self.semantic_fingerprint[i] = self.semantic_fingerprint.new_tensor(
            self._context_fingerprint(
                self._semantic_rows.get(identifier, {}), self.text_of(i),
                self._expectation_rows.get(identifier)))

    def occurrence_of(self, idx):
        """Stable identity; compaction changes a row index, not its occurrence."""
        if isinstance(idx, tuple) and idx[:1] == ('sym',):
            if not self.parts.contains(idx[1]) and (not callable(getattr(self, '_concept_point', None)) or self._concept_point(idx[1]) is None):
                raise IndexError('unavailable native entry')
            return idx
        i = int(idx)
        if not 0 <= i < len(self):
            raise IndexError(f"row {i} of {len(self)}")
        namespace = self.address_domain
        return ("ltm", namespace, int(self.address_keys[i]))

    @staticmethod
    def _role_occurrence_references(role_refs):
        """Return ordered grammatical occurrence edges from canonical roles.

        Integer-valued symbols remain ordinary opaque symbols.  Only the
        explicit, typed ``("ltm" | "thought", namespace, id)`` convention
        names an occurrence; role order (and repeated edges) is retained.
        """
        edges = []
        for role, reference in enumerate(role_refs):
            if not (isinstance(reference, tuple) and reference
                    and reference[0] in ('ltm', 'thought')):
                continue
            if (len(reference) != 3 or not isinstance(reference[1], str)
                    or type(reference[2]) is not int or reference[2] in (-1, 0)):
                raise ValueError("invalid semantic occurrence reference")
            edges.append((role, reference))
        return tuple(edges)

    @staticmethod
    def _occurrence_references(meaning):
        """The ordered compound-role edges of one complete meaning."""
        return LTM0._role_occurrence_references(meaning.role_refs)

    @staticmethod
    def _context_occurrence_references(context):
        """Find explicit occurrence addresses anywhere in semantic context.

        Bindings and scopes retain their addressed records but are not
        grammatical child edges, so they do not alter structure-read depth.
        Iteration avoids treating an arbitrary integer as an address and does
        not add a second reference index to the durable owner.
        """
        references, pending = [], [context]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
                continue
            if not isinstance(value, (tuple, list)):
                continue
            if value and value[0] in ('ltm', 'thought'):
                if (len(value) != 3 or not isinstance(value[1], str)
                        or type(value[2]) is not int or value[2] in (-1, 0)):
                    raise ValueError("invalid semantic context occurrence reference")
                references.append(tuple(value))
                continue
            pending.extend(value)
        return tuple(dict.fromkeys(references))

    @staticmethod
    def _validate_constituent_graph(contexts, namespace):
        """Validate local semantic links before replacing owner metadata.

        All local references must resolve among the proposed records.
        Bindings and scopes retain their records without hidden syntax. Cyclic references
        are legal. Readers terminate at repeated occurrences; neither a cycle
        nor a successful binding grants evidence to the referenced row.
        """
        for context in contexts.values():
            references = LTM0._context_occurrence_references(context)
            if any(ref[2] not in contexts
                   for ref in references
                   if ref[:2] == ('ltm', namespace)):
                raise ValueError("semantic context occurrence is unavailable")
    def _validate_local_references(self, meaning):
        """Reject dangling local references before append mutates any buffer."""
        namespace = self.address_domain
        references = [
            reference
            for reference in self._context_occurrence_references(meaning.metadata())
            if reference[:2] == ('ltm', namespace)
        ]
        if not references:
            return
        identifiers = set(int(identifier) for identifier
                          in self.address_keys[:len(self)].tolist())
        if any(reference[2] not in identifiers for reference in references):
            raise ValueError("semantic constituent occurrence is unavailable")

    def read_structure(self, reference, *, max_nodes=128, max_depth=16,
                       max_records=1024):
        """Read actual chain references; an idea has no stored derivation."""
        for name, value in (("max_nodes", max_nodes), ("max_depth", max_depth),
                            ("max_records", max_records)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if not isinstance(reference, tuple) or reference[:1] not in (('ltm',), ('sym',)):
            raise ValueError('structure requires an idea or chain reference')
        from ChainRead import envelope
        incomplete, occurrences, rows, edges, visited = [], [], [], [], set()
        pending, scanned = [(reference, 0)], 0
        while pending:
            current, depth = pending.pop()
            if current in visited:
                continue
            if depth > max_depth:
                incomplete.append('depth_limit'); continue
            if len(visited) >= max_nodes:
                incomplete.append('node_limit'); break
            if scanned >= max_records:
                incomplete.append('record_limit'); break
            index = self.location(current)
            if index is None:
                incomplete.append('unavailable_reference'); continue
            scanned += 1
            visited.add(current)
            value = {key: self._snapshot_tensor(item) if torch.is_tensor(item) else item
                     for key, item in self.row(index).items()}
            if value['meaning'] is None:
                incomplete.append('unavailable_metadata'); continue
            occurrences.append(current); rows.append(value)
            if isinstance(index, tuple):
                children = [(role, envelope(self, ref)) for role, ref in
                    enumerate(self.parts.references(index[1])) if ref[0] != SYMBOL]
                edges.extend((current, role, child) for role, child in children)
                pending.extend((child, depth + 1) for _, child in reversed(children))
        return dict(occurrences=tuple(occurrences), rows=tuple(rows), edges=tuple(edges),
                    records_scanned=scanned, incomplete=tuple(dict.fromkeys(incomplete)))

    def meaning_of(self, idx):
        """An owned detached value, or None when required context is missing."""
        value = self.raw_meaning_of(idx)
        return None if value is None else self.read_reference_bindings(idx, value)

    def raw_meaning_of(self, idx):
        """The retained concept and metadata, before working interpretation."""
        if isinstance(idx, tuple) and idx[:1] == ('sym',):
            return self.part_row(idx[1])['meaning']
        i = int(idx)
        if not 0 <= i < len(self):
            raise IndexError(f"row {i} of {len(self)}")
        identifier = int(self.address_keys[i])
        context = self._semantic_rows.get(identifier)
        if bool(self.metadata_required[i]) and context is None:
            return None
        from dataclasses import replace
        value = ConceptualMeaning.from_description(self.slots[i].detach())
        value = replace(value, polarity=self.polarity_of(i), sentence_kind='idea', **(context or {}))
        identity = dict(value.bindings).get('_identity')
        return replace(value, role_refs=(identity, None, None)) if identity is not None else value

    def semantic_extras(self):
        """Versioned context/provenance beside the existing tensor state.

        Role vectors and scalar evidence have one owner: registered buffers.
        This sidecar never duplicates them or supplies a second semantic store.
        """
        records = []
        for i in range(len(self)):
            identifier = int(self.address_keys[i])
            context = self._semantic_rows.get(identifier)
            if bool(self.metadata_required[i]) and context is None:
                raise ValueError("cannot checkpoint evidence with missing semantic metadata")
            records.append({"id": identifier, "context": dict(context or {}),
                            "text": self.text_of(i),
                            "expectation": self.expectation_of(i)})
        return {"version": 11, "document_sources": dict(self._document_sources),
                "supplies": copy.deepcopy(self.__dict__.get('_supplies', {})),
                "namespace": self.address_domain,
                "recurrence_identities": tuple(self._recurrence_identities),
                "cover_symbols": dict(self._cover_symbols),
                "records": records}

    def load_semantic_extras(self, extras):
        """Restore metadata only onto the matching durable occurrences."""
        if not isinstance(extras, dict) or extras.get('version') != 11:
            raise ValueError('unsupported LTM0 semantic checkpoint version')
        namespace = self.address_domain
        if extras.get('namespace') != namespace:
            raise ValueError('truth semantic checkpoint occurrence namespace differs')
        identities = extras.get('recurrence_identities')
        if (not isinstance(identities, (tuple, list)) or len(identities) != self.recurrence_capacity
                or any(ref is not None and not (
                    isinstance(ref, (tuple, list)) and (
                        len(ref) == 2 and ref[0] == 'sym' and type(ref[1]) is int and ref[1] > 0
                        or len(ref) == 3 and tuple(ref[:2]) == ('ltm', namespace)
                        and type(ref[2]) is int and ref[2] not in (-1, 0))) for ref in identities)):
            raise ValueError('invalid angular-recurrence identity checkpoint')
        covers = extras.get('cover_symbols', {})
        if (not isinstance(covers, dict) or any(not isinstance(key, str) or len(key) != 64
                or any(char not in '0123456789abcdef' for char in key)
                or type(symbol) is not int or symbol <= 0 for key, symbol in covers.items())
                or len(set(covers.values())) != len(covers)):
            raise ValueError('invalid document cover concept checkpoint')
        rows = {int(self.address_keys[i]): i for i in range(len(self))}
        if len(rows) != len(self):
            raise ValueError("duplicate truth occurrence identity")
        incoming, texts, expectation_rows = {}, [None] * len(self), {}
        for record in extras.get("records", ()):
            identifier = int(record["id"])
            if identifier in getattr(self, '_discarded_occurrences', ()):
                continue
            if identifier not in rows or identifier in incoming:
                raise ValueError(f"truth semantic checkpoint has an unavailable or duplicate occurrence {identifier}; "
                                 f"loaded addresses {tuple(rows)}")
            i = rows[identifier]
            context = record.get("context") or {}
            if not isinstance(context, dict) or set(context) - {"bindings", "scope"}:
                raise ValueError("invalid truth semantic context")
            if 'clause' in record:
                raise ValueError('a current truth checkpoint cannot contain a clause derivation')
            stored_context = dict(context)
            from dataclasses import replace
            meaning = replace(ConceptualMeaning.from_description(self.slots[i].detach()),
                polarity=self.polarity_of(i), **stored_context)
            incoming[identifier] = {key: meaning.metadata()[key]
                                    for key in ("bindings", "scope")}
            text = record.get("text")
            if text is not None and not isinstance(text, str):
                raise ValueError("truth source text must be a string")
            texts[i] = text
            if 'expectation' not in record:
                raise ValueError('truth semantic checkpoint omits expectation provenance')
            expectation = self._normalise_expectation_provenance(record['expectation'], namespace)
            if expectation is not None:
                expectation_rows[identifier] = expectation
            if (meaning.has_context or text is not None or expectation is not None) and not bool(self.metadata_required[i]):
                raise ValueError("truth context disagrees with its required-metadata flag")
            if record.get('definition') is not None:
                raise ValueError('retired definition rows cannot be loaded as LTM1')
            fingerprint_values = self._context_fingerprint(incoming[identifier], text,
                expectation)
            fingerprint = self.semantic_fingerprint.new_tensor(fingerprint_values)
            if not torch.equal(fingerprint, self.semantic_fingerprint[i]):
                raise ValueError("truth semantic content differs from its checkpoint fingerprint")
        if set(incoming) != set(rows):
            raise ValueError("truth semantic checkpoint omits stored occurrences")
        # Validate the complete replacement before publishing any context/text
        # to the owner.  A corrupt sidecar cannot leave a partly-restored graph.
        self._validate_constituent_graph(incoming, namespace)
        self._validate_expectation_links(expectation_rows)
        for identifier, provenance in expectation_rows.items():
            if provenance['kind'] == 'estimate' and '_expected_symbols' not in dict(incoming[identifier]['bindings']):
                raise ValueError('estimate checkpoint lacks its symbol-bank prior')
        sources = extras.get('document_sources', {})
        documents = {bytes(doc.tolist()).hex() for doc in self.document_keys[:len(self)]}
        if not isinstance(sources, dict) or not set(sources) <= documents or any(
                value not in (self.ORIGIN_USER, self.ORIGIN_PROVISIONED) for value in sources.values()):
            raise ValueError('invalid LTM0 source document in semantic checkpoint')
        if sources != self._document_sources:
            raise ValueError('source documents disagree with the LTM0 addressing checkpoint')
        self._semantic_rows = incoming
        self._texts = texts
        self._expectation_rows = expectation_rows
        self._cover_symbols = dict(covers)
        self._recurrence_identities[:] = [None if ref is None else tuple(ref) for ref in identities]
        self._supplies = copy.deepcopy(extras.get('supplies', {}))
        self.rebuild_addresses()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        unknown = [key[len(prefix):] for key in state_dict if key.startswith(prefix)
                   and '.' not in key[len(prefix):] and key[len(prefix):] not in self._buffers
                   and key[len(prefix):] not in self.addressing._buffers]
        if unknown:
            raise ValueError(f'retired or unsupported LTM0 checkpoint column: {unknown[0]}')
        if int(state_dict.get(prefix + 'truth_schema', -1)) != 14:
            raise ValueError('unsupported LTM0 layout checkpoint version; regenerate pre-layout memories')
        if int(state_dict.get(prefix + 'recurrence_version', -1)) != 2:
            raise ValueError('unsupported item 3.0 recurrenceCapacity checkpoint version')
        slots = state_dict.get(prefix + 'slots')
        if slots is None or slots.shape != self.slots.shape:
            raise ValueError('LTM0 checkpoint capacity/width differs from construction')
        if state_dict.get(prefix + 'addressing.stream', self.stream).shape != self.stream.shape:
            raise ValueError('LTM0 checkpoint stream capacity differs from construction')
        if int(state_dict.get(prefix + 'percept_layout', -1)) != self.percept_width:
            raise ValueError('LTM0 checkpoint percept layout differs from construction')
        for name, expected in self.named_buffers(recurse=False):
            if name in self._non_persistent_buffers_set:
                continue
            value = state_dict.get(prefix + name)
            if value is None or value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError('LTM0 layout checkpoint column differs from construction: ' + name)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError('nonfinite LTM0 checkpoint column: ' + name)
        for name, expected in self.addressing.named_buffers():
            value = state_dict.get(prefix + 'addressing.' + name)
            if value is None or value.shape != expected.shape or value.dtype != expected.dtype:
                raise ValueError('LTM0 layout checkpoint column differs from construction: ' + name)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError('nonfinite LTM0 checkpoint address index: ' + name)
        count = int(state_dict[prefix + 'count'])
        readings = state_dict[prefix + 'addressing.recurrence_readings']
        used = state_dict[prefix + 'addressing.recurrence_used']
        if (bool(((readings < -1) | (readings >= self.recurrence_capacity)).any())
                or bool((readings[count:] != -1).any())
                or not bool(used[readings[readings >= 0]].all())
                or bool((state_dict[prefix + 'addressing.recurrence_documents'] < 0).any())
                or bool((state_dict[prefix + 'addressing.recurrence_weights'] < 0).any())
                or bool(((state_dict[prefix + 'addressing.recurrence_symbols'] == 0)
                         | (state_dict[prefix + 'addressing.recurrence_symbols'] < -1)).any())):
            raise ValueError('invalid angular-recurrence checkpoint')
        keys = state_dict[prefix + 'addressing.address_keys'][:count].tolist()
        if not 0 <= count <= self.capacity or len(set(keys)) != count or any(key in (-1, 0) for key in keys):
            raise ValueError('invalid LTM0 checkpoint addresses/count')
        self._document_sources.clear()
        self._semantic_rows, self._expectation_rows, self._texts = {}, {}, []
        self._supplies = {}
        self._epoch_rows = None
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def accept_fact(self, idx):
        """Validate a TruthSet ending. Its input document and evidence carry admission."""
        i = int(idx)
        if not 0 <= i < len(self):
            raise IndexError(f"row {i} of {len(self)}")
        if self.kind_of(i) in ("question", "estimate", "cover"):
            raise ValueError("a question, estimate or cover cannot certify its own referent")

    def bind_constituents(self, meaning, *, reserve=0, stream=-1, parent_address=None):
        """Bind a pure compose tree through this existing occurrence owner.

        Preflight the entire graph and capacity before writing. Embedded
        clauses acquire no fact authority. The returned root keeps its live
        tensor path; only the ordinary child writes detach their payloads.
        """
        order, visited, active = [], set(), set()

        def local_refs(value):
            if isinstance(value, tuple):
                if value and value[0] == "constituent":
                    yield value
                else:
                    for item in value:
                        yield from local_refs(item)

        def visit(value, depth=0):
            if id(value) in active or depth > 64:
                raise ValueError("semantic constituent cycle or depth limit")
            if id(value) in visited:
                return
            if value.roles.shape[-1] != self.nDim:
                raise ValueError("meaning width differs from the truth store")
            self._validate_local_references(value)
            for field in value.metadata().values():
                for ref in local_refs(field):
                    if (len(ref) != 2 or type(ref[1]) is not int
                            or not 0 <= ref[1] < len(value.constituents)):
                        raise ValueError("semantic constituent reference is unavailable")
            active.add(id(value))
            for child in value.constituents:
                visit(child, depth + 1)
            active.remove(id(value))
            visited.add(id(value))
            order.append(value)

        visit(meaning)
        from Occurrence import native_content_key
        document = ('constituents', parent_address or native_content_key(meaning))
        child_addresses = {id(value): dict(document_key=document, sentence_index=ordinal,
            content_key=native_content_key(value)) for ordinal, value in enumerate(order[:-1])}
        new_children = sum(self._address_for_write(value, kind='unverified',
            **child_addresses[id(value)])[-1] is None for value in order[:-1])
        if new_children + reserve > self.capacity - len(self):
            raise OverflowError(f'LTM capacity {self.capacity} exhausted; forgetting is required')
        from ThoughtReferences import has_open_slots
        bound, references = {}, {}
        for value in order:
            def remap(item):
                if isinstance(item, tuple):
                    if item and item[0] == "constituent":
                        return references[id(value.constituents[item[1]])]
                    return tuple(remap(part) for part in item)
                return item
            result = ConceptualMeaning(value.roles, value.role_mask,
                **{key: remap(item) for key, item in value.metadata().items()})
            bound[id(value)] = result
            if value is not meaning:
                index = self.append_meaning(result,
                    kind="question" if has_open_slots(result) else "unverified",
                    stream=stream, **child_addresses[id(value)])
                if index < 0:  # preflight above reserves the complete graph
                    raise RuntimeError("constituent capacity changed during append")
                references[id(value)] = self.occurrence_of(index)
        return bound[id(meaning)]

    @torch.no_grad()
    def append_meaning(self, meaning, *, kind="fact",
                       trust=0.0, stream=-1, surprise=-1., evidence=None, order=0,
                       document_key=None, sentence_index=0, content_key=None, address=None, when=True):
        """Commit a complete description with explicit evidential provenance."""
        if not isinstance(meaning, ConceptualMeaning):
            raise TypeError("append_meaning requires a ConceptualMeaning")
        if meaning.roles.shape[-1] != self.nDim:
            raise ValueError("meaning width differs from the truth store")
        if kind not in self.KINDS:
            raise ValueError(f"unknown evidence kind {kind!r}")
        from ThoughtReferences import has_open_slots
        if kind == "fact" and has_open_slots(meaning):
            raise ValueError("a fact cannot have open references")
        if not math.isfinite(float(trust)):
            raise ValueError("fact trust must be finite")
        if type(stream) is not int or stream < -1:
            raise ValueError("index stream must be a row or shared (-1)")
        if stream >= self.stream.shape[1]:
            raise OverflowError('stream exceeds LTM0 construction capacity')
        surprise = float(surprise)
        if not math.isfinite(surprise) or not (surprise == -1. or 0. <= surprise <= 1.):
            raise ValueError("surprise must be unknown (-1) or in [0, 1]")
        if type(order) is not int or order < 0:
            raise ValueError('a stored field requires a nonnegative order')
        if evidence is None:
            signed = max(-1., min(1., float(trust)))
            evidence = (max(signed, 0.), max(-signed, 0.))
        else:
            evidence = tuple(map(float, evidence))
        if len(evidence) != 2 or any(not math.isfinite(x) or not 0 <= x <= 1 for x in evidence):
            raise ValueError('evidence poles must be finite and in [0, 1]')
        identifier, document, position, content, previous = self._address_for_write(
            meaning, kind=kind, document_key=document_key, sentence_index=sentence_index,
            content_key=content_key, key=address)
        if previous is None and len(self) >= self.capacity:
            raise OverflowError(f'LTM capacity {self.capacity} exhausted; forgetting is required')
        meaning = self.bind_constituents(meaning, reserve=int(previous is None), stream=stream,
                                         parent_address=(document, position, content))
        n = len(self) if previous is None else previous
        if previous is not None:
            evidence = (max(float(self.truth_plus[n]), evidence[0]), max(float(self.truth_minus[n]), evidence[1]))
        self._validate_local_references(meaning)
        if previous is not None and meaning.has_context:
            contexts = dict(self._semantic_rows)
            contexts[identifier] = meaning.metadata()
            self._validate_constituent_graph(contexts, self.address_domain)
        # Multiple working roles are not a durable layout. Estimates use their
        # unary envelope; other writers must supply the actual composed idea.
        occupied = meaning.role_mask.nonzero().flatten()
        if len(occupied) != 1 and kind != 'estimate':
            raise ValueError('LTM0 stores one idea; write semantic pairs through write_clause')
        from RowTime import read_stamp, stamp, write_stamp
        # Re-witnessing preserves the original event time, independently of
        # the content carrier supplied by this reading.
        original_stamp = read_stamp(self, n).clone() if previous is not None else None
        self.slots[n].copy_(meaning.roles[int(occupied[0])])
        self.truth_plus[n], self.truth_minus[n] = evidence
        self.surprise[n] = surprise
        self.address_keys[n] = identifier
        self.document_keys[n] = self.document_keys.new_tensor(list(document))
        self.sentence_content_keys[n] = self.sentence_content_keys.new_tensor(list(content))
        self.sentence_index[n] = position
        write_stamp(self, n, original_stamp if previous is not None else
                    stamp(self) if when is True else when)
        self.access_count[n] += 1
        context = {key: meaning.metadata()[key] for key in ('bindings', 'scope')}
        if meaning.role_refs[0] is not None:
            context['bindings'] = tuple(sorted(dict(context['bindings'], _identity=meaning.role_refs[0]).items()))
        self._semantic_rows[identifier] = context
        self._update_forward_index(identifier)
        self.metadata_required[n] = (bool(context['bindings'] or context['scope']) or (previous is not None and self.text_of(n) is not None)
                                     or identifier in self._expectation_rows)
        self._ensure_texts(n + 1)
        if previous is None:
            self._texts[n] = None
        self._update_semantic_fingerprint(n)
        self.witness_stream(n, stream)
        if previous is None:
            self.count.fill_(n + 1)
            start = getattr(self, '_epoch_rows', None)
            if start is not None:
                start.setdefault('initial_observations', {})[identifier] = (
                    int(self.access_count[n]), float(self.surprise[n]))
        self._index_occurrences[identifier] = n
        return n

    @torch.no_grad()
    def append_estimate(self, meaning, *, presence_logits, source_occurrences=(),
                        stream=None, document=None, trust=0.0):
        """Append one explicit prediction without granting fact authority.

        The estimate's entry vector lives in the ordinary idea row. Its
        symbol-bank prior, confidence, source occurrences and
        intended observation are sidecar provenance bound to that same row.
        No arriving observation is accepted here, so this method cannot make
        a prediction self-confirming. An estimate carries no time and
        consumes no arrival tick.
        """
        if not isinstance(meaning, ConceptualMeaning):
            raise TypeError("append_estimate requires a ConceptualMeaning")
        from Meaning import retained_symbol_prediction
        if retained_symbol_prediction(meaning) is None:
            raise ValueError('new estimates require a complete symbol-bank prior')
        logits = torch.as_tensor(presence_logits).detach().reshape(-1)
        if logits.numel() != 3 or not bool(torch.isfinite(logits).all()):
            raise ValueError("estimate requires three finite presence logits")
        namespace = self.address_domain
        provenance = self._normalise_expectation_provenance({
            "kind": "estimate",
            "source_occurrences": tuple(source_occurrences),
            "stream": stream,
            "document": document,
            "intended_occurrence": None,
            "presence_logits": tuple(float(value) for value in logits.to("cpu").tolist()),
        }, namespace)
        identifiers = set(int(identifier)
                          for identifier in self.address_keys[:len(self)].tolist())
        if any((not self.parts.contains(reference[1]) if reference[0] == 'sym'
                else reference[2] not in identifiers)
               for reference in self._expectation_references(provenance)):
            raise ValueError("estimate source occurrence is unavailable")
        index = self.append_meaning(
            meaning, kind="estimate", trust=trust, when=False,
            **self._estimate_write_address(meaning, document=document, stream=stream))
        # Re-witnessing ordinarily preserves an entry's original stamp;
        # an estimate is untimed even if an earlier witness carried one.
        from RowTime import write_stamp
        write_stamp(self, index, False)
        identifier = int(self.address_keys[index])
        self._expectation_rows[identifier] = provenance
        self.metadata_required[index] = True
        self._update_semantic_fingerprint(index)
        return index

    @torch.no_grad()
    def link_estimate_observation(self, estimate, observation):
        """Link an existing estimate to its later external observation.

        Only the relationship changes here; neither row's predicted nor
        observed meaning is rewritten after the target arrives.
        """
        if isinstance(observation, tuple) and observation[:1] == ('sym',):
            if not self.parts.contains(observation[1]):
                raise IndexError('expectation Part observation is unavailable')
            estimate_id = int(self.address_keys[int(estimate)])
            previous = self._expectation_rows.get(estimate_id)
            if previous is None or previous['kind'] != 'estimate':
                raise ValueError('expectation link must start at an estimate')
            if previous['intended_occurrence'] is not None:
                raise ValueError('estimate is already linked to an observation')
            self._expectation_rows[estimate_id] = dict(previous, intended_occurrence=observation)
            self._update_semantic_fingerprint(estimate)
            self._validate_expectation_links()
            return
        estimate, observation = int(estimate), int(observation)
        if not 0 <= estimate < len(self) or not 0 <= observation < len(self):
            raise IndexError("expectation link row is unavailable")
        if self.kind_of(estimate) != "estimate":
            raise ValueError("expectation link must start at an estimate")
        if self.kind_of(observation) not in ("observation", "question", "fact", "unverified"):
            raise ValueError("expectation link must end at an external observation")
        estimate_id = int(self.address_keys[estimate])
        observation_id = int(self.address_keys[observation])
        previous = self._expectation_rows.get(estimate_id)
        if previous is None or previous["kind"] != "estimate":
            raise ValueError("estimate lacks retained provenance")
        if previous["intended_occurrence"] is not None:
            raise ValueError("estimate is already linked to an observation")
        namespace = self.address_domain
        observation_ref = ("ltm", namespace, observation_id)
        estimate_ref = ("ltm", namespace, estimate_id)
        linked_estimate = dict(previous)
        linked_estimate["intended_occurrence"] = observation_ref
        linked_estimate = self._normalise_expectation_provenance(
            linked_estimate, namespace)
        prior = self._expectation_rows.get(observation_id)
        history = () if prior is None else self._expectation_references(prior)
        linked_observation = self._normalise_expectation_provenance({
            "kind": "observation", "estimate_occurrence": estimate_ref,
            **({'previous_estimates': history} if history else {}),
        }, namespace)
        self._expectation_rows[estimate_id] = linked_estimate
        self._expectation_rows[observation_id] = linked_observation
        self.metadata_required[estimate] = True
        self.metadata_required[observation] = True
        self._semantic_rows.setdefault(estimate_id, {})
        self._semantic_rows.setdefault(observation_id, {})
        self._ensure_texts(len(self))
        self._update_semantic_fingerprint(estimate)
        self._update_semantic_fingerprint(observation)
        self._validate_expectation_links()

    @torch.no_grad()
    def append_expectation_pair(self, estimate, observation, *, presence_logits,
                                source_occurrences=(), stream=None, document=None,
                                observation_kind="observation", trust=0.0, index_stream=-1):
        """Retain or re-witness both addresses; insufficient capacity raises."""
        if observation_kind not in ("observation", "question"):
            raise ValueError("expectation pair requires an observed input kind")
        from Meaning import retained_symbol_prediction, retained_symbol_activation, ClosingImage, withheld_symbols, symbol_identities
        from Expectation import retention_surprise
        prediction = retained_symbol_prediction(estimate)
        if prediction is None:
            raise ValueError('new estimates require a complete symbol-bank prior')
        from dataclasses import replace
        packet = MeaningExpectation(estimate.roles, torch.as_tensor(presence_logits).to(estimate.roles),
            estimate.bindings, estimate.scope, prediction.symbol_keys, prediction.symbol_logits,
            prediction.symbol_ids, prediction.symbol_orders)
        estimate = replace(estimate, bindings=packet.retained_bindings(
            observed=observation.roles, occupied=observation.role_mask,
            activation=retained_symbol_activation(observation, prediction),
            identities=symbol_identities(observation), key_valid=prediction.key_valid))
        prediction = retained_symbol_prediction(estimate)
        view = ClosingImage.form(observation.roles, estimate.roles,
            prediction.confidence, prediction=prediction,
            occupied=observation.role_mask, identities=symbol_identities(observation), withheld=withheld_symbols(observation, prediction),
            activation=retained_symbol_activation(observation, prediction))
        surprise = retention_surprise(view.excess-view.omission, view.expected)
        old = self._address_for_write(observation, kind=observation_kind)[-1]
        old_estimate = self._address_for_write(estimate, kind='estimate',
            **self._estimate_write_address(estimate, document=document, stream=stream))[-1]
        required = int(old is None) + int(old_estimate is None)
        if self.capacity - len(self) < required:
            raise OverflowError(f'LTM capacity {self.capacity} exhausted; forgetting is required')
        observation = self.bind_constituents(observation, reserve=required, stream=index_stream)
        estimate_index = self.append_estimate(
            estimate, presence_logits=presence_logits,
            source_occurrences=source_occurrences, stream=stream,
            document=document)
        if estimate_index < 0:
            # Capacity was checked above; keep the ordinary write fail-loud
            # if a future store implementation violates that invariant.
            raise RuntimeError("estimate append unexpectedly exhausted durable capacity")
        observation_index = self.append_meaning(
            observation, kind=observation_kind, trust=trust,
                stream=index_stream, surprise=surprise)
        if observation_index < 0:
            raise RuntimeError("observation append unexpectedly exhausted durable capacity")
        self.link_estimate_observation(estimate_index, observation_index)
        return estimate_index, observation_index

    def expectation_of(self, idx):
        """Return an owned immutable-value copy of a row's expectation link."""
        if isinstance(idx, tuple) and idx[:1] == ('sym',):
            if not self.parts.contains(idx[1]):
                raise IndexError('unavailable Part observation')
            refs = tuple(('ltm', self.address_domain, identifier)
                for identifier, value in self._expectation_rows.items()
                if value['kind'] == 'estimate' and value['intended_occurrence'] == idx)
            return (None if not refs else dict(kind='observation',
                estimate_occurrence=refs[-1], previous_estimates=refs[:-1]))
        i = int(idx)
        if not 0 <= i < len(self):
            raise IndexError(f"row {i} of {len(self)}")
        value = self._expectation_rows.get(int(self.address_keys[i]))
        return None if value is None else dict(value)

    def expectation_pair(self, idx, *, gain=1., object_mask=None):
        """Read one linked estimate/observation pair and derive its residual.

        The residual is intentionally reconstructed from the two retained
        meanings and stored confidence rather than held as a second mutable
        delta.  This keeps the estimate and actual observation authoritative
        and makes any mismatch auditable after replay.
        """
        i = idx if isinstance(idx, tuple) else int(idx)
        provenance = self.expectation_of(i)
        if provenance is None:
            raise ValueError("row has no retained expectation provenance")
        if provenance["kind"] == "estimate":
            estimate_index = i
            observation_ref = provenance["intended_occurrence"]
            if observation_ref is None:
                raise ValueError("estimate has no observed counterpart")
        else:
            observation_ref = self.occurrence_of(i)
            estimate_ref = provenance["estimate_occurrence"]
            estimate_index = self._index_occurrences.get(estimate_ref, -1)
            if estimate_index < 0:
                raise ValueError("observation estimate counterpart is unavailable")
        # Stable occurrence -> physical row is maintained by the existing
        # leaf index on append, restore and compaction; pair reads stay O(1).
        observation_index = self.location(observation_ref)
        if observation_index is None:
            raise ValueError("estimate observation counterpart is unavailable")
        estimate = self.raw_meaning_of(estimate_index)
        observation = self.raw_meaning_of(observation_index)
        if estimate is None or observation is None:
            raise ValueError("expectation pair has unavailable semantic metadata")
        if '_expectation_observation' in dict(estimate.bindings):
            from dataclasses import replace
            recorded = dict(estimate.bindings).get('_expectation_observation')
            observation = replace(observation, bindings=dict(observation.bindings, **dict(recorded)))
        estimate_provenance = self.expectation_of(estimate_index)
        if estimate_provenance is None or estimate_provenance["kind"] != "estimate":
            raise ValueError("expectation pair lacks estimate provenance")
        logits = estimate.roles.new_tensor(estimate_provenance["presence_logits"])
        observed_mask = observation.role_mask.to(device=estimate.roles.device)
        observed_roles = observation.roles.to(estimate.roles)
        from Meaning import ClosingImage, retained_symbol_prediction, retained_symbol_activation, withheld_symbols, symbol_identities
        prediction = retained_symbol_prediction(estimate)
        if prediction is None:
            raise ValueError('retained estimate lacks its symbol-bank prior')
        view = ClosingImage.form(observed_roles, estimate.roles, prediction.confidence,
            gain=gain, object_mask=object_mask, prediction=prediction, occupied=observed_mask, identities=symbol_identities(observation),
            activation=retained_symbol_activation(observation, prediction),
            withheld=None if prediction is None else withheld_symbols(observation, prediction))
        residual = view.excess-view.omission
        conceived, image = view.conceived, view.image
        return {
            "estimate": estimate,
            "observation": observation,
            "estimate_occurrence": self.occurrence_of(estimate_index),
            "observation_occurrence": self.occurrence_of(observation_index),
            "source_occurrences": estimate_provenance["source_occurrences"],
            "stream": estimate_provenance["stream"],
            "document": estimate_provenance["document"],
            "intended_occurrence": estimate_provenance["intended_occurrence"],
            "presence_logits": logits,
            "residual": residual,
            "conceived": conceived,
            "gain": view.gain, "expected_activation": view.expected,
            "excess": view.excess, "omission": view.omission, "identities": view.identities,
            "content": conceived,
            "surprise": float(self.parts.surprise[observation_index[1]]
                if isinstance(observation_index, tuple) else self.surprise[observation_index]),
        }

    def __len__(self):
        return int(self.count.item())

    def _fit(self, vec):
        """Conform an idea operand to the store width (leading ``nDim``,
        zero-padded). ``None`` -> the zero (Null) vector."""
        if vec is None:
            return self.slots.new_zeros(self.nDim)
        v = vec.reshape(-1).to(self.slots)
        if v.numel() == self.nDim:
            return v
        out = self.slots.new_zeros(self.nDim)
        k = min(self.nDim, v.numel())
        out[:k] = v[:k]
        return out

    @torch.no_grad()
    def append_idea(self, point, *, trust=0.0, kind="fact", evidence=None, order=0, **address):
        return self.append_meaning(ConceptualMeaning.from_description(self._fit(point)),
            trust=trust, kind=kind, evidence=evidence, order=order, **address)

    @torch.no_grad()
    def append_relation(self, part, whole, *, truth_plus=0., truth_minus=0., when=None, surprise=-1.):
        return ('sym', self.parts.add(part, whole, truth_plus=truth_plus,
            truth_minus=truth_minus, when=when, surprise=surprise))

    def row(self, idx: int) -> dict:
        """Read slots, evidence, stable provenance and an owned meaning.

        Legacy np1/vp/np2 tensors remain views; ``meaning`` is an owned detached
        value, or None when required semantic metadata is unavailable.
        """
        if isinstance(idx, tuple) and idx[:1] == ('sym',):
            return self.part_row(idx[1])
        n = int(self.count.item())
        if not (0 <= int(idx) < n):
            raise IndexError(f"row {idx} of {n}")
        i = int(idx)
        return {
            'point': self.slots[i],
            'stream': tuple(self.stream[i].nonzero().flatten().tolist()),
            'trust': float(self.trust[i].item()),
            'order': self.order_of_row(i),
            'evidence': (float(self.truth_plus[i]), float(self.truth_minus[i])),
            'corners': self.corners(i),
            'row_id': int(self.address_keys[i]),
            'when': self.entry_when(i).clone(),
            'surprise': float(self.surprise[i].item()),
            'origin': self.source_document(i),
            'text': self.text_of(i),
            'kind': self.kind_of(i),
            'pending': bool(dict(self._semantic_rows.get(int(self.address_keys[i]), {})
                                .get('bindings', ())).get('_pending', False)),
            'occurrence': self.occurrence_of(i),
            'meaning': self.meaning_of(i),
            'expectation': self.expectation_of(i),
            'metadata_complete': (not bool(self.metadata_required[i])
                                  or int(self.address_keys[i]) in self._semantic_rows),
        }

    def relations(self, reading=None):
        """Native Part symbols, optionally selected by their arguments' types."""
        rows = self.parts.indices()
        if reading is None or reading == 'part':
            return rows
        required = {'implies': 'conditional', 'equal': 'equality',
                    'definition': 'definition', 'attribution': 'attribution',
                    'subsumption': 'subsumption'}[reading]
        return rows.new_tensor([row for row in rows.tolist()
                                if required in self.part_row(row)['readings']])

    def ideas(self):
        return torch.arange(len(self), device=self.slots.device)

    @property
    def trust(self):
        """Signed view of the two independent evidence lanes."""
        return self.truth_plus - self.truth_minus

    @torch.no_grad()
    def set_trust(self, idx: int, trust: float):
        """Set signed evidence through its two poles (verification/reassertion)."""
        i = int(idx)
        if not (0 <= i < int(self.count.item())):
            raise IndexError(f"row {i} of {int(self.count.item())}")
        if not math.isfinite(float(trust)):
            raise ValueError("fact trust must be finite")
        signed = max(-1.0, min(1.0, float(trust)))
        self.truth_plus[i], self.truth_minus[i] = max(signed, 0.), max(-signed, 0.)

    @torch.no_grad()
    def set_evidence(self, idx, positive, negative):
        """Replace the two independent evidence poles, retaining provenance."""
        values = (float(positive), float(negative))
        if any(not math.isfinite(x) or not 0. <= x <= 1. for x in values):
            raise ValueError("evidence poles must be finite and in [0, 1]")
        if isinstance(idx, tuple) and idx[:1] == ('sym',):
            i = int(idx[1])
            if not self.parts.contains(i):
                raise IndexError(f"unavailable Part {i}")
            self.parts.truth_plus[i] = max(float(self.parts.truth_plus[i]), values[0])
            self.parts.truth_minus[i] = max(float(self.parts.truth_minus[i]), values[1])
            self.parts.refresh_effective()
            if bool(self.parts.ref_kinds[i].eq(self.parts.SYMBOL).all()):
                self.parts._symbol_epoch += 1
            return
        i = int(idx)
        if not 0 <= i < len(self):
            raise IndexError(f"row {i} of {len(self)}")
        self.truth_plus[i], self.truth_minus[i] = values

    def corners(self, idx):
        """Compute four overlapping corners without collapsing the poles."""
        p, n = float(self.truth_plus[idx]), float(self.truth_minus[idx])
        return min(p, 1-n), min(n, 1-p), min(p, n), min(1-p, 1-n)

    # -- Provenance ------------------------------------------------------

    def _ensure_texts(self, n: int):
        """Pad the host-side ``_texts`` list to ``n`` entries."""
        if len(self._texts) < n:
            self._texts.extend([None] * (n - len(self._texts)))

    def text_of(self, idx: int):
        """The row's source text, or None (unset / post-load)."""
        i = int(idx)
        return self._texts[i] if 0 <= i < len(self._texts) else None

    @torch.no_grad()
    def set_origin(self, idx: int, origin: int, text=None, *, asserted=None):
        """Tag a row's writer provenance (``ORIGIN_*``) and optionally
        attach its source text -- retroactive, mirroring ``set_trust``,
        because provisioned / user rows land via the observe-site push
        during the forward and are identified only afterwards."""
        i = int(idx)
        n = int(self.count.item())
        if not (0 <= i < n):
            raise IndexError(f"row {i} of {n}")
        if (text is not None and bool(self.metadata_required[i])
                and int(self.address_keys[i]) not in self._semantic_rows):
            raise ValueError("cannot update source text with missing semantic context")
        if origin not in (self.ORIGIN_CONVERSATION, self.ORIGIN_PROVISIONED, self.ORIGIN_USER):
            raise ValueError('unknown LTM0 source document authority')
        key = bytes(self.document_keys[i].tolist()).hex()
        if self._document_sources.get(key, self.ORIGIN_CONVERSATION) != origin:
            sources = dict(self._document_sources)
            if origin == self.ORIGIN_CONVERSATION:
                sources.pop(key, None)
            else:
                sources[key] = int(origin)
            self._replace_document_sources(sources)
        self._ensure_texts(n)
        if text is not None:
            self._texts[i] = str(text)
            if not bool(self.metadata_required[i]):
                self._semantic_rows.setdefault(int(self.address_keys[i]), {})
            self.metadata_required[i] = True
            self._update_semantic_fingerprint(i)
        if origin in (self.ORIGIN_PROVISIONED, self.ORIGIN_USER):
            from PresentationMemory import supplied_occurrence
            supplied_occurrence(self, i, asserted=asserted)

    def rows_of_origin(self, *origins):
        """Row indices whose ``origin`` is any of ``origins``, in row
        (append) order."""
        c = int(self.count.item())
        if c == 0 or not origins:
            return self.count.new_zeros(0)
        og = self.count.new_tensor([self.source_document(i) for i in range(c)])
        mask = torch.zeros_like(og, dtype=torch.bool)
        for o in origins:
            mask |= og == int(o)
        return mask.nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def withdraw_origin(self, origin: int, *, preserve_addresses=()) -> None:
        """Withdraw one writer's authority without deleting its content.

        Tensor-only checkpoint restore uses this intermediate state while
        semantic sidecars and thought-owned roots are unavailable.  Accepted
        facts become unverified; questions/estimates retain their provenance;
        all selected rows lose trust.
        """
        count = len(self)
        selected = self.count.new_zeros(count,dtype=torch.bool)
        selected[self.rows_of_origin(origin)] = True
        for key in preserve_addresses:
            selected &= self.address_keys[:count] != int(key)
        self.truth_plus[:count].masked_fill_(selected, 0)
        self.truth_minus[:count].masked_fill_(selected, 0)
        from PresentationMemory import withdraw_supplies
        withdraw_supplies(self, origin=origin, preserve=preserve_addresses)

    @torch.no_grad()
    def clear_origin(self, origin: int, *, retained_occurrences=(), preserve_addresses=()) -> int:
        """Withdraw an origin and compact only its unreachable records.

        The survivor closure starts from rows of other origins and explicit
        roots supplied by another live owner (ordinary thought history).  A
        retained withdrawn row stays at its stable occurrence ID with its
        slots, context and source text, but no longer certifies a fact.  The
        full closure is resolved and validated before any evidence or buffer is
        changed.  IDs and logical clocks are never rewound.
        """
        c = int(self.count.item())
        if c == 0:
            self.withdraw_origin(origin, preserve_addresses=preserve_addresses)
            return 0
        selected = self.count.new_zeros(c,dtype=torch.bool)
        selected[self.rows_of_origin(origin)] = True
        for key in preserve_addresses:
            selected &= self.address_keys[:c] != int(key)
        if not bool(selected.any()):
            self.withdraw_origin(origin, preserve_addresses=preserve_addresses)
            return 0
        namespace = self.address_domain
        indices = {int(self.address_keys[i]): i for i in range(c)}
        if len(indices) != c:
            raise ValueError("duplicate truth occurrence identity")

        keep_rows = set((~selected).nonzero(as_tuple=True)[0].tolist())
        pending = list(keep_rows)
        for reference in retained_occurrences:
            if (not isinstance(reference, tuple) or len(reference) != 3
                    or reference[:2] != ('ltm', namespace)
                    or type(reference[2]) is not int
                    or reference[2] not in indices):
                raise ValueError("retained occurrence reference is unavailable")
            pending.append(indices[reference[2]])

        seen = set()
        while pending:
            index = pending.pop()
            if index in seen:
                continue
            seen.add(index)
            keep_rows.add(index)
            meaning = self.meaning_of(index)
            if meaning is None:
                raise ValueError(
                    "cannot retain constituents with unavailable semantic metadata")
            for reference in self._context_occurrence_references(meaning.metadata()):
                if reference[:2] != ('ltm', namespace):
                    continue
                child = indices.get(reference[2])
                if child is None:
                    raise ValueError("retained constituent occurrence is unavailable")
                pending.append(child)
            provenance = self._expectation_rows.get(
                int(self.address_keys[index]))
            for reference in self._expectation_references(provenance):
                child = indices.get(reference[2])
                if child is None:
                    raise ValueError("retained expectation occurrence is unavailable")
                pending.append(child)

        # This includes all closure members (including context-only roots), so
        # validation cannot accidentally erase an invalid survivor's evidence.
        self._validate_constituent_graph(
            {
                int(self.address_keys[index]): self._semantic_rows.get(
                    int(self.address_keys[index]), {})
                for index in keep_rows
            },
            namespace,
        )
        keep = self.count.new_tensor(sorted(keep_rows))
        n_keep = int(keep.numel())
        removed = c - n_keep
        # Only after every resolution/validation succeeds may evidence change.
        self.withdraw_origin(origin, preserve_addresses=preserve_addresses)
        if removed == 0:
            return 0
        self.compact_rows(keep)
        return removed

    row_columns = ('slots', 'surprise', 'address_keys', 'document_keys',
                   'sentence_content_keys', 'sentence_index', 'access_count',
                   'metadata_required', 'semantic_fingerprint', 'truth_plus', 'truth_minus',
                   'stream', 'recurrence_readings')

    @torch.no_grad()
    def compact_rows(self, keep):
        """Compact physical rows in store order; stable addresses do not move."""
        count = len(self)
        keep = torch.as_tensor(keep, dtype=torch.long, device=self.count.device)
        rows = keep.tolist()
        if rows != sorted(set(rows)) or any(i < 0 or i >= count for i in rows):
            raise ValueError('LTM compaction requires ordered, unique surviving rows')
        size = len(rows)
        from PresentationMemory import withdraw_supplies
        survivors = set(rows)
        withdraw_supplies(self, removed=[int(self.address_keys[i]) for i in range(count) if i not in survivors])
        owner_ref = self.__dict__.get('_concept_owner')
        owner = None if owner_ref is None else owner_ref()
        if owner is not None:
            survivors = set(rows)
            for index in range(count):
                if index not in survivors:
                    owner.mtm.forget(self.occurrence_of(index))
        self._ensure_texts(count)
        texts = [self._texts[i] for i in rows]
        minus_one = {'surprise', 'address_keys', 'sentence_index', 'recurrence_readings'}
        for name in self.row_columns:
            values = getattr(self, name)
            values[:size] = values[keep]
            values[size:count].fill_(-1 if name in minus_one else 0)
        alive = set(map(int, self.address_keys[:size].tolist()))
        for name in ('_semantic_rows', '_expectation_rows'):
            setattr(self, name, {key: value for key, value in getattr(self, name).items() if key in alive})
        self._texts = texts + [None] * (len(self._texts) - size)
        live_documents = {bytes(row.tolist()).hex() for row in self.document_keys[:size]}
        self._replace_document_sources({k:v for k,v in self._document_sources.items() if k in live_documents})
        self.count.fill_(size)
        self.rebuild_addresses()
        self._validate_expectation_links()
        rebuild = getattr(self, '_rebuild_taxonomy', None)
        if rebuild is not None:
            rebuild()

    @torch.no_grad()
    def reset(self, *, keep_learned_readings=False):
        """Clear the store (idempotent)."""
        self._supplies = {}
        self.slots.zero_()
        self.stream.zero_()
        self.surprise.fill_(-1.)
        self.count.zero_()
        self._texts = []
        self.address_keys.fill_(-1)
        self.document_keys.zero_()
        self.sentence_content_keys.zero_()
        self.sentence_index.fill_(-1)
        self.access_count.zero_()
        self.metadata_required.zero_()
        self.semantic_fingerprint.zero_()
        self._semantic_rows = {}
        self._expectation_rows = {}
        self.clear_recurrence(keep_learned=keep_learned_readings)
        for name in ('forget_passes', 'forget_sentences', 'forget_last_sentence', 'forget_cutoff', 'forget_pending'):
            getattr(self, name).zero_()
        self.truth_plus.zero_()
        self.truth_minus.zero_()
        self._replace_document_sources({})
        self._index_occurrences = AddressIndex()

        # Re-reading the same source after reset recomputes the same address. A held reference
        # cannot silently bind to a later fact at the same physical row.


class LTM1(MemoryStore, nn.Module):
    OCCURRENCE, SYMBOL, RELATION = OCCURRENCE, SYMBOL, RELATION
    columns = ('refs', 'ref_kinds', 'truth_plus', 'truth_minus', 'access_count', 'surprise', 'percept', 'effective_plus', 'effective_minus', 'operator', 'grammar_operation')

    def __init__(self, capacity=8192):
        super().__init__()
        self.capacity = int(capacity)
        if self.capacity < 2:
            raise ValueError('LTM1 capacity must include a reserved zero and a relation seat')
        self.allocate = None
        self.handle_reserved = None
        # One pair and two percept bands (where + when, four coordinates each)
        # per declared relation seat; there is no growth on load.
        self.register_buffer('refs', torch.full((self.capacity, 2), -1, dtype=torch.long))
        self.register_buffer('ref_kinds', torch.zeros(self.capacity, 2, dtype=torch.int8))
        self.register_buffer('percept', torch.zeros(self.capacity, 8))
        self.register_buffer('truth_plus', torch.zeros(self.capacity))
        self.register_buffer('truth_minus', torch.zeros(self.capacity))
        self.register_buffer('effective_plus', torch.zeros(self.capacity))
        self.register_buffer('effective_minus', torch.zeros(self.capacity))
        self.register_buffer('operator', torch.full((self.capacity,), -1, dtype=torch.long))
        self.register_buffer('grammar_operation', torch.full((self.capacity,), -1, dtype=torch.long))
        self.operation_catalog = ()
        from TrustGates import TrustGates
        self.gates = TrustGates()
        self._occupation_owner = None
        self._refreshing_effective = False
        self._primitive_evidence = set()
        self.register_buffer('relation_order', torch.full((self.capacity,), -1, dtype=torch.long))
        self._order_revision = None
        self._effective_source = None
        self._concept_owner = None
        self.register_buffer('access_count', torch.zeros(self.capacity, dtype=torch.long))
        self.register_buffer('surprise', torch.full((self.capacity,), -1.))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('layout_version', torch.tensor(VERSION, dtype=torch.long))
        self._index_version = None
        self._symbol_epoch = 0
        self.epoch_start = None
        self.epoch_handles = set()
        self._vacant_epoch_handles = []
        self._pairs, self._parts, self._wholes, self._references = {}, {}, {}, {}

    def _reserved_handle(self, row):
        if self.handle_reserved is not None and self.handle_reserved(row):
            return True
        if self.rows_referencing(SYMBOL, row):
            return True
        store_ref = getattr(self, '_occurrence_store', None)
        store = None if store_ref is None else store_ref()
        if store is None:
            return False
        if row in store._cover_symbols.values() or bool((store.recurrence_symbols == row).any()):
            return True
        names = (('sym', row), ('name', row))
        return any(ref in names for context in store._semantic_rows.values()
                   for ref in (*context.get('role_refs', ()), dict(context.get('bindings', ())).get('_identity')))

    @torch.no_grad()
    def begin_epoch(self):
        """Restore the starting rows, including access counts and both evidence lanes."""
        if self.epoch_start is None:
            self.epoch_start = {name: self._snapshot_tensor(value, cpu=True)
                                for name, value in self.state_dict().items()}
            return set()
        before = set(self.indices().tolist())
        state = {name: self.epoch_start[name] for name in self.state_dict()}
        # Learned readiness is monotone across epoch row restoration.
        state['_extra_state'] = self.get_extra_state()
        self.load_state_dict(state)
        self.epoch_start.pop('_initial_observations', None)
        occupied = set(self.indices().tolist())
        # Only handles reserved by this relation allocator are available again.
        # Explicit symbol handles belong to their concepts across epochs.
        self._vacant_epoch_handles = sorted(self.epoch_handles - occupied, reverse=True)
        self._symbol_epoch += 1
        self._index_version = None
        self._index()
        return before - occupied

    def __len__(self):
        return int(self.count)

    @property
    def revision(self):
        return (id(self.refs), self.refs._version, self.ref_kinds._version,
                self.truth_plus._version, self.truth_minus._version, self.access_count._version, len(self))

    @property
    def _topology_revision(self):
        return (id(self.refs), self.refs._version, self.ref_kinds._version, len(self))

    @staticmethod
    def ordered_rows(refs, kinds):
        """Topological order for a live slab or an epoch snapshot."""
        occupied = set(kinds[:, 0].ne(0).nonzero().flatten().tolist())
        visiting, visited, ordered = set(), set(), []
        for root in sorted(occupied):
            pending = [(root, False)]
            while pending:
                row, finished = pending.pop()
                if finished:
                    visiting.remove(row)
                    visited.add(row)
                    ordered.append(row)
                elif row in visiting:
                    raise ValueError('cyclic relation reference')
                elif row not in visited:
                    if row not in occupied:
                        raise ValueError('dangling relation reference')
                    visiting.add(row)
                    pending.append((row, True))
                    pending.extend((value, False) for kind, value in
                                   zip(kinds[row].tolist(), refs[row].tolist()) if kind == RELATION)
        return ordered

    @torch.no_grad()
    def _sort(self):
        """Keep the constructed order slab in dependency order; handles stay stable."""
        if self._order_revision == self._topology_revision:
            return
        ordered = self.ordered_rows(self.refs, self.ref_kinds)
        self.relation_order.fill_(-1)
        if ordered:
            self.relation_order[:len(ordered)] = self.relation_order.new_tensor(ordered)
        self._order_revision = self._topology_revision

    def configure_operations(self, occupations):
        import weakref
        if occupations is None or not occupations.operations:
            return
        catalogue = occupations.operations
        if self.operation_catalog and catalogue != self.operation_catalog:
            raise ValueError('relation operation catalogue cannot change after attachment')
        self.operation_catalog = catalogue
        self._occupation_owner = weakref.ref(occupations)
        self._effective_source = None

    def get_extra_state(self):
        from IndependentComponents import _pack_state
        return _pack_state(dict(operation_catalog=self.operation_catalog, gate_open=dict(self.gates.open)))

    @torch.no_grad()
    def clear_operation(self, row):
        """Naming makes a live pair independent of learned-operation authority."""
        if not self.contains(row):
            raise IndexError('unavailable relation')
        self.grammar_operation[row] = -1
        self._effective_source = None

    def _check_extra_state(self, value):
        from IndependentComponents import _unpack_state
        state = _unpack_state(value)
        if not isinstance(state, dict) or 'gate_open' not in state:
            raise ValueError('LTM1 checkpoint requires gate state; regenerate pre-gate memories')
        if tuple(state['operation_catalog']) != self.operation_catalog:
            raise ValueError('relation operation catalogue does not match the loaded grammar')
        return self.gates.validate_state(state['gate_open'])

    def set_extra_state(self, value):
        self.gates.restore(self._check_extra_state(value))
        self._effective_source = None

    def gate_measurements(self):
        """Report readiness; only refresh_effective can open a gate."""
        self.refresh_effective()
        owner = None if self._concept_owner is None else self._concept_owner()
        occupations = None if self._occupation_owner is None else self._occupation_owner()
        values = self.gates.measurements(owner, occupations)
        result = {name: dict(open=self.gates.open[name], measurement=values[name]) for name in self.gates.names}
        if occupations is not None and 'lift' in occupations.operations:
            result['clause']['arguments'] = occupations.operation('lift')['argument_measurements']
        return result

    @torch.no_grad()
    def refresh_effective(self):
        """Propagate changed concept and operation authority in dependency order."""
        if self._refreshing_effective:
            return
        self._sort()
        owner = None if self._concept_owner is None else self._concept_owner()
        trust = None if owner is None else owner.similarity_codebook.trust
        occupations = None if self._occupation_owner is None else self._occupation_owner()
        revision = (self._topology_revision, self.truth_plus._version, self.truth_minus._version,
                    self.operator._version, self.grammar_operation._version,
                    None if occupations is None else occupations.revision,
                    None if trust is None else trust.revision, self.gates.revision)
        if revision == self._effective_source:
            return
        from ConceptTrust import positive_length
        def concept(identity):
            return 1. if owner is None else positive_length(owner, identity)
        def endpoint(kind, value):
            if kind == RELATION:
                return max(float(self.effective_plus[value]), float(self.effective_minus[value]))
            if kind == SYMBOL and owner is not None and value in owner._concept_allocator.placement:
                return concept(value)
            return 1.
        def write(row, operation_trust=1.):
            scale = min(endpoint(*ref) for ref in self.references(row))
            operator = int(self.operator[row])
            if operator != -1:
                scale = min(scale, concept(operator))
            scale = min(scale, operation_trust)
            left, right = self.references(row)
            named = (owner is not None and left[0] == right[0] == SYMBOL
                     and left[1] in owner._concept_allocator.named_identities.get(right[1], ()))
            operation = int(self.grammar_operation[row])
            name = None if operation == -1 else self.operation_catalog[operation]
            scale = min(scale, self.gates.cap(named=named, operation=name))
            self.effective_plus[row] = self.truth_plus[row] * scale
            self.effective_minus[row] = self.truth_minus[row] * scale
        rows = self.relation_order[:len(self)].tolist()
        # Occupation distributions use denotation to select an order. Native
        # named/membership pairs have no grammar factor: make those independent
        # authorities available before asking the categorizer for its factors.
        from PartsOfSpeech import FIXED_OPERATIONS
        self._primitive_evidence = {row for row in rows
            if (int(self.grammar_operation[row]) == -1 or
                self.operation_catalog[int(self.grammar_operation[row])] in FIXED_OPERATIONS)
            and all(kind != RELATION for kind, _ in self.references(row))}
        self._refreshing_effective = True
        try:
            for row in self._primitive_evidence:
                write(row)
            # Empty banks keep their declared start. Opening is an authority
            # read over live rows, never a construction-time side effect.
            if rows and not all(self.gates.open.values()):
                for name in self.gates.names:
                    if self.gates.open[name]:
                        continue
                    measured = self.gates.measurements(owner, occupations)[name]
                    if self.gates.advance(name, measured):
                        self._effective_source = None
                        # A newly admitted name can change the categorizer's
                        # order selection for the next measurement/factor.
                        for row in self._primitive_evidence:
                            write(row)
            factors = {}
            for row in rows:
                operation = int(self.grammar_operation[row])
                if operation != -1 and operation not in factors:
                    if occupations is None:
                        raise ValueError('relation operation has no loaded occupation owner')
                    factors[operation] = occupations.operation(self.operation_catalog[operation])['trust']
            for row in rows:
                if row not in self._primitive_evidence:
                    write(row, factors.get(int(self.grammar_operation[row]), 1.))
        finally:
            self._refreshing_effective = False
        self._effective_source = (*revision[:-2], None if trust is None else trust.revision, self.gates.revision)

    @property
    def effective_revision(self):
        self.refresh_effective()
        return (id(self), self.effective_plus._version, self.effective_minus._version)

    def evidence(self, row):
        """The single authority read: effective support and counter-evidence."""
        if not self.contains(row):
            raise IndexError('unavailable relation')
        if self._refreshing_effective and row not in self._primitive_evidence:
            raise ValueError('operation occupation depends cyclically on learned relation authority')
        self.refresh_effective()
        return self.effective_plus[row], self.effective_minus[row]

    @property
    def symbol_revision(self):
        """Symbol topology and evidence; chain appends cannot expire membership."""
        return (id(self), self._symbol_epoch)

    @staticmethod
    def reference(value, kind=SYMBOL):
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError('a relation reference requires its tag and value')
            kind, value = value
        kind, value = int(kind), int(value)
        if kind not in (OCCURRENCE, SYMBOL, RELATION) or value in (-1, 0) or (kind != OCCURRENCE and value < 0):
            raise ValueError('invalid tagged relation reference')
        return kind, value

    def references(self, row):
        if not self.contains(row):
            raise IndexError('unavailable relation')
        return tuple(zip(self.ref_kinds[int(row)].tolist(), self.refs[int(row)].tolist()))

    def _index(self):
        version = (id(self.refs), self.refs._version, self.ref_kinds._version, self.operator._version, len(self))
        if self._index_version == version:
            return
        pairs, parts, wholes, refs = {}, {}, {}, {}
        for row in self.indices().tolist():
            left, right = (self.reference(ref) for ref in self.references(row))
            if (left, right) in pairs:
                raise ValueError('duplicate tagged relation')
            pairs[left, right] = row
            parts.setdefault(right, []).append(left)
            wholes.setdefault(left, []).append(right)
            dependencies = {left, right}
            if int(self.operator[row]) != -1:
                dependencies.add((SYMBOL, int(self.operator[row])))
            for ref in dependencies:
                refs.setdefault(ref, []).append(row)
        self._pairs, self._parts, self._wholes, self._references = pairs, parts, wholes, refs
        self._index_version = version

    def indices(self):
        return self.ref_kinds[:, 0].ne(0).nonzero().flatten()

    def contains(self, row):
        return 0 < int(row) < self.capacity and bool(self.ref_kinds[int(row)].ne(0).all())

    def find(self, part, whole, *, ref_kinds=(SYMBOL, SYMBOL)):
        self._index()
        return self._pairs.get((self.reference(part, ref_kinds[0]), self.reference(whole, ref_kinds[1])))

    def parts(self, whole):
        self._index()
        values = tuple(self._parts.get(self.reference(whole), ()))
        return values if isinstance(whole, tuple) else tuple(v for k, v in values if k == SYMBOL)

    def wholes(self, part):
        self._index()
        values = tuple(self._wholes.get(self.reference(part), ()))
        return values if isinstance(part, tuple) else tuple(v for k, v in values if k == SYMBOL)

    def rows_referencing(self, kind, value):
        self._index()
        return tuple(self._references.get(self.reference(value, kind), ()))

    def stratum(self, row):
        return 2 if any(kind == RELATION for kind, _ in self.references(row)) else 1

    def protected_references(self, rows=None):
        """All dependencies, transitively; freeze this set before forgetting."""
        pending = list(self.indices().tolist() if rows is None else rows)
        seen, references = set(), set()
        while pending:
            row = pending.pop()
            if row in seen:
                continue
            seen.add(row)
            if int(self.operator[row]) != -1:
                references.add((SYMBOL, int(self.operator[row])))
            for reference in self.references(row):
                references.add(reference)
                if reference[0] == RELATION:
                    if not self.contains(reference[1]):
                        raise ValueError('dangling relation reference')
                    pending.append(reference[1])
        return references

    @torch.no_grad()
    def add(self, part, whole, *, ref_kinds=(SYMBOL, SYMBOL), truth_plus=0., truth_minus=0.,
            when=None, where=None, surprise=-1., witness=True, symbol=None,
            operator=None, grammar_operation=None):
        left, right = self.reference(part, ref_kinds[0]), self.reference(whole, ref_kinds[1])
        if (SYMBOL, NULL_CONCEPT) in (left, right):
            raise ValueError('a null-concept cannot be a relation endpoint')
        if any(not math.isfinite(float(x)) or not 0 <= float(x) <= 1 for x in (truth_plus, truth_minus)):
            raise ValueError('relation evidence has two independent poles in [0, 1]')
        if operator is not None and (type(operator) is not int or operator == 0 or operator < -1):
            raise ValueError('relation operator must be a word identity or -1')
        if grammar_operation is not None and (type(grammar_operation) is not int
                or not -1 <= grammar_operation < len(self.operation_catalog)):
            raise ValueError('relation operation must index the loaded catalogue or be -1')
        if not math.isfinite(float(surprise)) or not (surprise == -1. or 0. <= surprise <= 1.):
            raise ValueError('relation surprise must be unknown (-1) or in [0, 1]')
        bands = {}
        for name, value in (('when', when), ('where', where)):
            if value is not None:
                value = torch.as_tensor(value, device=self.percept.device, dtype=self.percept.dtype)
                if value.shape != (4,) or not bool(torch.isfinite(value).all()):
                    raise ValueError('percept bands require four finite coordinates')
                bands[name] = value
        for kind, value in (left, right):
            if kind == RELATION and not self.contains(value):
                raise ValueError('relation child must exist before its parent')
        row = self.find(left, right)
        self._sort()
        new_row = row is None
        if row is not None and operator is not None and int(self.operator[row]) != operator:
            raise ValueError('one tagged relation cannot have different lexical operators')
        # The first composition owns this provenance tag. Later closes merge
        # their poles without replacing that origin (including a cleared tag).
        if row is None:
            if symbol is not None:
                row = int(symbol)
            elif self.allocate is None:
                free = self.ref_kinds[:, 0].eq(0).nonzero().flatten()
                free = free[free > 0]
                row = next((cid for cid in free.tolist() if not self._reserved_handle(cid)), None)
                if row is None:
                    raise OverflowError('LTM1 capacity exhausted; forgetting is required')
            else:
                # A name or seat can arrive after the reset or checkpoint.
                # Test its live owners at the actual point of reuse.
                while self._vacant_epoch_handles:
                    candidate = self._vacant_epoch_handles[-1]
                    if not self.contains(candidate) and not self._reserved_handle(candidate):
                        break
                    self._vacant_epoch_handles.pop()
                row = (self._vacant_epoch_handles.pop() if self._vacant_epoch_handles
                       else int(self.allocate()))
                self.epoch_handles.add(row)
            if not 0 < row < self.capacity:
                raise OverflowError('LTM1 handle exceeds its constructed capacity')
            if self.contains(row):
                raise RuntimeError('a live relation cannot be reassigned')
            self.refs[row] = self.refs.new_tensor((left[1], right[1]))
            self.ref_kinds[row] = self.ref_kinds.new_tensor((left[0], right[0]))
            self.relation_order[len(self)] = row
            self.count += 1
            self._order_revision = self._topology_revision
            self.operator[row] = -1 if operator is None else operator
            self.grammar_operation[row] = -1 if grammar_operation is None else grammar_operation
            if right[0] == SYMBOL and (left[0] == SYMBOL or right[1] in getattr(self, '_class_rows', ())):
                self._symbol_epoch += 1
        elif symbol is not None and int(symbol) != row:
            raise ValueError('a tagged pair already has a different handle')
        positive = max(float(self.truth_plus[row]), float(truth_plus))
        negative = max(float(self.truth_minus[row]), float(truth_minus))
        if (right[0] == SYMBOL and (left[0] == SYMBOL or right[1] in getattr(self, '_class_rows', ())) and
                (positive != float(self.truth_plus[row]) or negative != float(self.truth_minus[row]))):
            self._symbol_epoch += 1
        self.truth_plus[row] = positive
        self.truth_minus[row] = negative
        self.access_count[row] += int(witness)
        if surprise >= 0:
            self.surprise[row] = float(surprise)
        for name, value in bands.items():
            self.percept[row, 4:].copy_(value) if name == 'when' else self.percept[row, :4].copy_(value)
        if new_row and self.epoch_start is not None:
            self.epoch_start.setdefault('_initial_observations', {})[row] = (
                int(self.access_count[row]), float(self.surprise[row]))
        return row

    @torch.no_grad()
    def retain(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.refs.device)
        selected = set(indices.tolist())
        if len(selected) != len(indices) or any(not self.contains(i) for i in selected):
            raise ValueError('invalid LTM1 retention selection')
        if any(kind == RELATION and child not in selected for row in selected for kind, child in self.references(row)):
            raise ValueError('cannot discard a retained relation dependency')
        erase = torch.ones(self.capacity, dtype=torch.bool, device=self.refs.device)
        erase[indices] = False
        if bool((erase & self.ref_kinds[:, 1].eq(SYMBOL)).any()):
            self._symbol_epoch += 1
        for name in self.columns:
            getattr(self, name)[erase] = -1 if name in ('refs', 'surprise', 'operator', 'grammar_operation') else 0
        self.count.fill_(len(indices))
        self._sort()
        self.refresh_effective()

    def remove(self, part, whole):
        row = self.find(part, whole)
        if row is not None:
            self.retain([i for i in self.indices().tolist() if i != row])

    def remove_symbol(self, symbol):
        removed = set(self.rows_referencing(SYMBOL, symbol))
        if self.contains(symbol):
            removed.add(int(symbol))
        pending = list(removed)
        while pending:
            for parent in self.rows_referencing(RELATION, pending.pop()):
                if parent not in removed:
                    removed.add(parent); pending.append(parent)
        self.retain([i for i in self.indices().tolist() if i not in removed])

    def row(self, index):
        if not self.contains(index):
            raise IndexError('unavailable relation')
        return {name: getattr(self, name)[int(index)] for name in self.columns}

    def readings(self, index, types):
        left_ref, right_ref = self.references(index)
        left, right = set(types(left_ref)), set(types(right_ref))
        readings = []
        if 'situation' in left and 'situation' in right:
            readings.append('conditional')
        if 'situation' in left and 'speech_individual' in right:
            readings.append('attribution')
        if left_ref != right_ref and 'object' in left and 'word' in right:
            readings.append('definition')
        if 'kind' in left and 'kind' in right:
            readings.append('subsumption')
        if 'singleton' in right and 'definition' not in readings:
            readings.append('equality')
        if 'spatial' in left and 'spatial' in right:
            readings.append('part')
        return tuple(readings)

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        for name in ('layout_version', 'count'):
            value = state.get(prefix + name)
            if value is None or value.shape != () or value.dtype != torch.long:
                raise ValueError('invalid relation checkpoint version or count')
        if int(state.get(prefix + 'layout_version', -1)) != VERSION or prefix + 'ref_kinds' not in state:
            raise ValueError('unsupported LTM1 layout checkpoint version; regenerate pre-layout memories')
        if prefix + '_extra_state' not in state:
            raise ValueError('LTM1 checkpoint requires gate state; regenerate pre-gate memories')
        self._check_extra_state(state[prefix + '_extra_state'])
        refs, kinds = state.get(prefix + 'refs'), state[prefix + 'ref_kinds']
        if refs is None or refs.shape != self.refs.shape or kinds.shape != self.ref_kinds.shape:
            raise ValueError('LTM1 checkpoint capacity differs from construction')
        if kinds.dtype != torch.int8 or refs.dtype != torch.long:
            raise ValueError('invalid relation reference dtypes')
        occupied = kinds.ne(0).any(-1)
        valid = ((kinds >= OCCURRENCE) & (kinds <= RELATION)).all(-1)
        if bool((occupied & ~valid).any()) or bool((~occupied & refs.ne(-1).any(-1)).any()):
            raise ValueError('invalid restored reference tags')
        seen = set()
        for row in occupied.nonzero().flatten().tolist():
            if row == 0:
                raise ValueError('zero is not a relation handle')
            pair = tuple(self.reference((int(k), int(v))) for k, v in zip(kinds[row], refs[row]))
            if pair in seen:
                raise ValueError('duplicate restored tagged relation')
            seen.add(pair)
            if any(k == RELATION and (v >= self.capacity or not bool(occupied[v])) for k, v in pair):
                raise ValueError('dangling restored relation reference')
        if int(occupied.sum()) != int(state.get(prefix + 'count', -1)):
            raise ValueError('relation count disagrees with occupied rows')
        visiting, visited = set(), set()
        for root in occupied.nonzero().flatten().tolist():
            pending = [(root, False)]
            while pending:
                row, finished = pending.pop()
                if finished:
                    visiting.remove(row)
                    visited.add(row)
                elif row in visiting:
                    raise ValueError('cyclic restored relation reference')
                elif row not in visited:
                    visiting.add(row)
                    pending.append((row, True))
                    pending.extend((value, False) for kind, value in
                                   zip(kinds[row].tolist(), refs[row].tolist()) if kind == RELATION)
        for name in self.columns:
            value = state.get(prefix + name)
            if value is None or value.shape != getattr(self, name).shape or value.dtype != getattr(self, name).dtype:
                raise ValueError('relation checkpoint column differs from construction')
        for name in ('truth_plus', 'truth_minus', 'effective_plus', 'effective_minus'):
            values = state[prefix + name]
            if not bool(torch.isfinite(values).all() & (values >= 0).all() & (values <= 1).all()):
                raise ValueError('invalid restored relation evidence')
        if state[prefix + 'percept'].shape != self.percept.shape or not bool(torch.isfinite(state[prefix + 'percept']).all()):
            raise ValueError('invalid restored relation percept')
        surprise = state[prefix + 'surprise']
        if not bool(torch.isfinite(surprise).all() & ((surprise == -1) | ((surprise >= 0) & (surprise <= 1))).all()):
            raise ValueError('invalid restored relation surprise')
        if bool((state[prefix + 'access_count'] < 0).any()):
            raise ValueError('invalid restored relation access count')
        operators = state[prefix + 'operator']
        if bool(((operators < -1) | (operators == 0)).any()):
            raise ValueError('invalid restored relation operator')
        operations = state.get(prefix + 'grammar_operation')
        if (operations is None or operations.shape != self.grammar_operation.shape or operations.dtype != torch.long
                or bool(((operations < -1) | (operations >= len(self.operation_catalog))).any())):
            raise ValueError('invalid restored grammar operation catalogue index')
        order = state.get(prefix + 'relation_order')
        if order is None or order.shape != self.relation_order.shape or order.dtype != torch.long:
            raise ValueError('relation order differs from construction')
        count = int(occupied.sum())
        ordered = order[:count].tolist()
        if set(ordered) != set(occupied.nonzero().flatten().tolist()) or len(set(ordered)) != count or bool(order[count:].ne(-1).any()):
            raise ValueError('invalid restored relation order')
        before = set()
        for row in ordered:
            if any(k == RELATION and v not in before for k, v in zip(kinds[row].tolist(), refs[row].tolist())):
                raise ValueError('restored relation order must put parts before wholes')
            before.add(row)
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
        self._order_revision = self._topology_revision
        self._effective_source = None
        self._index_version = None
        self._symbol_epoch += 1
        self._index()
