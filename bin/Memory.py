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
VERSION = 6
# Reserved outside both address inventories. It never consumes a learnable seat.
NULL_CONCEPT = torch.iinfo(torch.int64).max


from Layers import Layer, MeaningExpectation
from util import TheXMLConfig
from Meaning import ConceptualMeaning
from ClauseRow import ClauseRows
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

    def __deepcopy__(self, memo):
        import weakref
        result = object.__new__(type(self))
        memo[id(self)] = result
        result.__dict__.update({name: copy.deepcopy(value, memo)
                               for name, value in self.__dict__.items()
                               if name not in ('_owner', '_time_owner')})
        for name in ('_owner', '_time_owner'):
            reference = self.__dict__.get(name)
            owner = None if reference is None else reference()
            if owner is not None:
                # A cloned history and its thought clock belong to the
                # cloned model, even though the live links are weak.
                setattr(result, name, weakref.ref(copy.deepcopy(owner, memo)))
        return result

    @staticmethod
    def _snapshot_tensor(value, *, detach=True, cpu=False):
        value = value.detach() if detach else value
        if cpu:
            value = value.cpu()
        return value.clone()

    @torch.no_grad()
    def restore_runtime_state(self, state):
        """Roll back admitted tensors without importing a durable checkpoint.

        A trial snapshot belongs to this construction. Migration, source
        validation and checkpoint warnings belong only to checkpoint loads.
        Validate the complete shape before changing any live buffer.
        """
        current = self.state_dict()
        if current.keys() != state.keys() or any(
                not torch.is_tensor(state[name]) or value.shape != state[name].shape
                or value.dtype != state[name].dtype for name, value in current.items()):
            raise ValueError('runtime memory snapshot differs from its construction')
        for name, value in current.items():
            value.copy_(state[name].to(value.device))
        self.clear_caches()


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
        from ThoughtStrata import construct
        object.__setattr__(self, '_thought_stores', construct(batch, self._init_capacity, concept_dim))

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
        if hard and hasattr(self, 'strata'):
            self.strata.clear(b)
        if hard:
            contents = list(self._thought_contents)
            for row in range(self._init_batch) if b is None else (b,):
                for store in self._thought_stores[row]:
                    store.expire()
                # Epoch time can repeat in a different chunk seat. Expired
                # views must leave with their stores, not match that new tick.
                contents[row] = (None,) * self._init_capacity
            self._thought_contents = tuple(contents)
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
    modifiers: tuple = ()
    modal_taught: bool = False
    case_value: torch.Tensor | None = None
    verb_meaning: torch.Tensor | None = None


class Timeline(NamedTuple):
    events: tuple


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
    def events(self):
        return self.timelines[0].events

    def with_actual(self, *, events=None, **metadata):
        actual = Timeline(self.events if events is None else events)
        return replace(self, timelines=(actual, *self.timelines[1:]), **metadata)


class MediumTermMemory(MemoryStore):
    """References form a path across kept closes; values are computed on read.

    The pool owns membership and recorded action operands, never a second
    copy of an idea's content. Entry addresses survive physical slot reuse.
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
        self.pending = {}
        self.pending_step = 0
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

    def hold_pending(self,reference,meaning):
        marks=dict(meaning.bindings)
        identity = marks.get('_identity')
        named = isinstance(identity, tuple) and identity[:1] == ('name',)
        if not (marks.get('_open_references') or marks.get('_forward_references')
                or marks.get('_unresolved_operands') or named):
            return
        # Pending grammar is working memory; completed provenance is not.
        names=('_open_references','_forward_references','_reference_sites',
               '_bound_roles','_bound_references','_unresolved_operands')
        if named:
            names = (*names, '_identity')
        marks={name:marks[name] for name in names if name in marks}
        if reference not in self.pending and len(self.pending)>=self.capacity:
            del self.pending[min(self.pending,key=lambda key:self.pending[key][0])]
        from Meaning import ConceptualMeaning
        def owned(value):
            # A thought's constituent can be a view of an expiring STM
            # store. Pending grammar owns its values, not that store or
            # the operator modules of a trial's extension tree.
            return ConceptualMeaning(value.roles.detach().clone(),
                value.role_mask.detach().clone(), **value.metadata(),
                constituents=tuple(owned(child) for child in value.constituents))
        self.pending[reference]=(self.pending_step,replace(owned(meaning),bindings=marks))

    def pending_meaning(self,reference,value):
        pending=self.pending.get(reference)
        return value if pending is None else pending[1]

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
            if entry.refinement.get('member_kind') == 'percept':
                # Basic percepts are members of an order-zero concept through
                # its feature table. They are not events or relation rows.
                row = owner._csw_row_of(symbol)
                for _, reference in entry.refinement['members']:
                    if reference[:1] != ('bp',):
                        raise ValueError('a percept collection requires basic percept references')
                    owner.add_concept_feature(row, 'bp', reference[1], 1.)
                return
            from MemoryPaths import membership, member
            order = entry.refinement.get('member_order')
            # Re-entering an already known individual does not make that
            # individual a higher-order class of its own references.
            reentry = order is not None and owner._concept_source_order(symbol) == order
            if not reentry:
                for occurrence in entry.occurrences:
                    membership(owner, occurrence, int(symbol))
            from FamiliarTier import class_members
            alloc = owner._concept_allocator
            affected = {parent for occurrence in entry.occurrences
                        for kind, parent in alloc.parts.wholes(member(owner._closed_clause_store(), occurrence)[0])
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
            latest = max(entry.when for entry in self.entries.values())
            def recency(key):
                uses = self.accesses.get(key, {})
                current = any(tick > self.closes.get(stream, 0) for stream, tick in uses.items())
                return current, max(uses.values(), default=-1), self.entries[key].when - latest
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
        self.entries[expected] = Token((), None, 1, int(when), (Timeline(()),),
            transient=self._allocate_id(), pending=True, where=self._where(None), order=order,
            support=support, mass=sum(support.values()))
        return expected

    def match_candidate(self, order, value):
        value = self._value(value)
        scored = [(float(torch.nn.functional.cosine_similarity(self.value_of(key).to(value), value, dim=0)), key)
                  for key, entry in self.entries.items() if entry.order == order
                  and entry.symbol is None and 'promoted_percept' not in entry.refinement]
        score, key = max(scored, key=lambda item: item[0], default=(-1., None))
        return key if score >= self.match_cos else None

    def update_candidate(self, key, *, when, value, support):
        entry = self.entries[key]
        if entry.order is None:
            raise ValueError('candidate update requires a candidate')
        self._value(value)
        support = self._support(support)
        if not support:
            raise ValueError('candidate update requires support')
        mass = sum(support.values())
        merged = dict(entry.support)
        for row, weight in support.items():
            merged[row] = merged.get(row, 0.) + weight
        self.entries[key] = replace(entry, count=entry.count+1, when=int(when),
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

    def read_member(self, address, concept=None, *, live=False):
        """Read an entry and its denoted concept; MTM owns neither value."""
        reference = getattr(self, '_owner', None)
        owner = None if reference is None else reference()
        if owner is None:
            raise ValueError('MTM reference read requires its STM/LTM owner')
        from MemberRead import read
        value = read(owner, address, concept)
        return value if live else self._value(value)

    def event_value(self, event, *, live=False):
        return self.read_member(event.occurrence, live=live)

    def member_references(self, reference, *, timeline=0):
        key = None if isinstance(reference, Token) else self.token_of(reference)
        if key is None and not isinstance(reference, Token):
            return ()
        entry = reference if isinstance(reference, Token) else self.entries[key]
        if not 0 <= timeline < len(entry.timelines):
            raise ValueError(f'identity {key!r} has no timeline {timeline}')
        concepts = entry.refinement.get('member_concepts', {})
        members = list(entry.refinement.get('members', ()))
        covered = {address for address, concept in members}
        members.extend((address, concepts.get(address, self.reading_of(address)))
                       for address in entry.occurrences if address not in covered)
        if not members and key is not None and key[:1] == ('name',):
            members.append((key, key[1]))
        members.extend((event.occurrence, None) for event in entry.timelines[timeline].events
                       if event.role == 0 and event.before is not None)
        members.extend((('sym', cid), cid) for cid in dict.fromkeys(
            cid for cid in (entry.native, entry.symbol) if cid is not None))
        return tuple(dict.fromkeys(members))

    def value_of(self, reference, *, timeline=0, live_candidate=False, references=None):
        """Compute the collection from its member references on every read."""
        token = None if isinstance(reference, Token) else self.token_of(reference)
        if token is None and not isinstance(reference, Token):
            return None
        entry = reference if isinstance(reference, Token) else self.entries[token]
        if entry.order is not None:
            owner = self._owner()
            from FamiliarTier import centroid
            value = centroid(owner, entry.support)
            return value if live_candidate else self._value(value)
        members = self.member_references(entry if token is None else token, timeline=timeline)
        # A participation witness without forward operands records an event,
        # but does not replace its participant's state with the whole clause.
        events = [event for event in entry.timelines[timeline].events
                  if event.role == 0 and event.before is not None]
        if events:
            latest = events[-1]
            clock = entry.refinement.get('event_links', {}).get(latest.occurrence)
            later = () if clock is None or timeline != 0 else tuple((address, self.reading_of(address))
                for address in entry.occurrences
                if entry.refinement.get('mentions', {}).get(address, (None, -1))[1] > clock[1])
            members = ((latest.occurrence, None), *later)
        if references is not None and not events:
            members = tuple(dict.fromkeys((*references, *members)))
        if not members:
            raise ValueError(f'identity {token!r} has no member references')
        values = [self.read_member(address, concept, live=live_candidate)
                  for address, concept in members]
        if any(value.shape != values[0].shape for value in values):
            raise ValueError(f'identity {token!r} references have different widths')
        return torch.stack(values).amax(0)

    def record_link(self, reference, occurrence, stream, *, kind=None, order=None):
        """The member's path clock is the current kept close, not world time."""
        key = self.token_of(reference)
        if key is None:
            raise ValueError('link clock requires a retained collection')
        entry = self.entries[key]
        refinement = dict(entry.refinement)
        mentions = dict(refinement.get('mentions', {}))
        mentions[occurrence] = (stream, self.closes.get(stream, 0)+1)
        refinement['mentions'] = mentions
        if kind is not None:
            if refinement.get('member_kind', kind) != kind:
                raise ValueError('a collection cannot mix member kinds')
            refinement.setdefault('member_kind', kind)
        if order is not None:
            if refinement.get('member_order', order) != order:
                raise ValueError('a collection cannot mix member orders')
            refinement.setdefault('member_order', order)
        self.entries[key] = replace(entry, refinement=refinement)
        self.keep_access(key, stream)
        return key

    def in_window(self, reference, stream):
        key = self.token_of(reference)
        if key is None:
            return False
        mentions = self.entries[key].refinement.get('mentions', {})
        latest = max((clock for member_stream, clock in mentions.values()
                      if member_stream == stream), default=None)
        return latest is not None and self.closes.get(stream, 0)+1-latest <= self.horizon

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
                when=when,
                pending=entry.pending and pending, where=self._where(where, entry.where))
            self.names[occurrence] = Name(occurrence, reading)
            return occurrence
        self._room(retain)
        identity = self._allocate_id(transient)
        self.entries[occurrence] = Token((occurrence,), kind, 1, when, (Timeline(()),),
            native=native, pending=pending, transient=identity, where=self._where(where))
        self.names[occurrence] = Name(occurrence, reading)
        return occurrence

    def append_path(self, key, occurrence, *, kind, when, where, stream, order, concept=None):
        """Collections may share a member; a name still has one referent."""
        if key is None:
            key = ('collection', kind, occurrence, concept)
            if key not in self.entries:
                self._room()
                self.entries[key] = Token((), None, 0, when, (Timeline(()),),
                    transient=self._allocate_id(), where=self._where(where))
        entry = self.entries[key]
        members = entry.occurrences if occurrence in entry.occurrences else (*entry.occurrences, occurrence)
        self.entries[key] = replace(entry, occurrences=members, count=entry.count+1,
            when=when, where=self._where(where, entry.where))
        return self.record_link(key, occurrence, stream, kind=kind, order=order)

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
            (Timeline(()),), pending=True,
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
        if value is not None:
            self._value(value)
        self.entries[token] = entry.with_actual(occurrences=uses,
            kind=kind if entry.kind is None else entry.kind,
            pending=entry.pending and kind in (None, NULL_CONCEPT),
            count=entry.count + 1, when=when, where=self._where(where, entry.where))
        self.names[occurrence] = Name(token, reading)
        return token

    def alternative(self, token, *, when, before=None):
        """Continue a reference path from the actual prefix, without a value copy."""
        key = self.token_of(token)
        if key is None:
            raise ValueError('an alternative requires a retained identity')
        entry = self.entries[key]
        if len(entry.timelines) == 1:
            self.entries[key] = replace(entry, when=when,
                timelines=(*entry.timelines, Timeline(entry.events)))
        return len(self.entries[key].timelines)-1



    def event(self, token, occurrence, *, verb, role, when, before=None, basis=None,
              modifiers=(), timeline=0, case_value=None, verb_meaning=None):
        key = self.token_of(token)
        if key is None or role not in (0, 2) or type(verb) is not int or verb <= 0:
            raise ValueError('an event requires a retained identity, verb and participant role')
        entry = self.entries[key]
        if type(timeline) is not int or not 0 <= timeline < len(entry.timelines):
            raise ValueError('an event requires a retained timeline')
        old = entry.timelines[timeline]
        if case_value is not None and (self._value(case_value).shape != self.value_of(key).shape or
                                     bool(((case_value < 0) | (case_value > 1)).any())):
            raise ValueError('an event case test requires its native content shape and chart')
        event = Event(occurrence, verb, role, when,
            None if before is None else self._value(before),
            self._value(basis) if torch.is_tensor(basis) else basis,
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
        refinement = dict(entry.refinement)
        stream = next(reversed(self.accesses.get(key, {})), None)
        links = dict(refinement.get('event_links', {}))
        links[occurrence] = (stream, self.closes.get(stream, 0)+1)
        refinement['event_links'] = links
        self.entries[key] = replace(entry, when=when, refinement=refinement, timelines=tuple(
            Timeline(events if index == timeline else item.events)
            for index, item in enumerate(entry.timelines)))

    def _detach(self, occurrence):
        """Detach a mention; an event with this address has its own owner."""
        if occurrence in self.entries and (self.entries[occurrence].order is not None or occurrence[:1] == ('collection',)):
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
            uses = tuple(member for member in entry.occurrences if member != occurrence)
            if token[:1] == ('collection',) and not uses:
                self._release_id(entry)
                self.entries.pop(token)
                self.accesses.pop(token, None)
                continue
            refinement = dict(entry.refinement)
            refinement['members'] = tuple(member for member in refinement.get('members', ()) if member[0] != occurrence)
            for name in ('mentions', 'event_links', 'member_concepts'):
                if name in refinement:
                    refinement[name] = {key: value for key, value in refinement[name].items() if key != occurrence}
            self.entries[token] = replace(entry, occurrences=uses, refinement=refinement, timelines=tuple(
                Timeline(tuple(event for event in timeline.events if event.occurrence != occurrence))
                for timeline in entry.timelines))

    def merge(self, token, into, *, when):
        """Join a pending object without moving the antecedent's existing uses."""
        source, target = self.token_of(token), self.token_of(into)
        if source is None or target is None:
            raise ValueError('MTM merge requires two retained tokens')
        if source == target:
            return target
        previous, antecedent = self.entries[source], self.entries[target]
        for field in ('member_kind', 'member_order'):
            left, right = previous.refinement.get(field), antecedent.refinement.get(field)
            if left is not None and right is not None and left != right:
                raise ValueError(f'a merge cannot mix collection kinds or orders: {field} {left!r} != {right!r}')
        if previous.refinement.get('member_kind') in ('state', 'event', 'percept'):
            from IdentityRows import meet_ratios
            left, right = self.value_of(source), self.value_of(target)
            covered, covers = meet_ratios(left, right)
            if not bool((covered == 1) | (covers == 1)):
                raise ValueError('a merge requires containment between its collections')
        mentions = {**previous.refinement.get('mentions', {}), **antecedent.refinement.get('mentions', {})}
        members = tuple(dict.fromkeys((*antecedent.occurrences, *previous.occurrences)))
        members = tuple(sorted(members, key=lambda member: mentions.get(member, (None, -1))[1]))
        self.entries[target] = antecedent.with_actual(
            occurrences=members,
            count=antecedent.count + previous.count, when=when,
            kind=previous.kind if antecedent.kind is None else antecedent.kind,
            pending=antecedent.pending and previous.pending,
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
        refinement['members'] = tuple(dict.fromkeys((*previous.refinement.get('members', ()), *refinement.get('members', ()))))
        for field in ('member_concepts', 'event_links'):
            refinement[field] = {**previous.refinement.get(field, {}), **refinement.get(field, {})}
        self.entries[target] = replace(self.entries[target], refinement=refinement)
        for stream, clock in self.accesses.pop(source, {}).items():
            retained = self.accesses.setdefault(target, {})
            retained[stream] = max(clock, retained.get(stream, 0))
        self._release_id(previous)
        del self.entries[source]
        return target

    def close(self, store, *, stream=None, references=()):
        """Advance once after a kept sentence; staging and reads never call this."""
        self.pending_step += 1
        self.pending={key:value for key,value in self.pending.items()
            if self.pending_step-value[0] <= self.horizon}
        clock = self.closes.get(stream, 0) + 1
        self.closes[stream] = clock
        for key, entry in self.entries.items():
            if entry.order is None:
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

    def state(self):
        """Checkpoint only committed values; export owns every tensor copy."""
        def snapshot(value):
            return self._snapshot_tensor(value, cpu=True) if torch.is_tensor(value) else value
        def event_state(event):
            return (event.occurrence, event.verb, event.role, event.when,
                snapshot(event.before), snapshot(event.basis), event.taught,
                tuple(map(snapshot, event.modifiers)), event.modal_taught,
                snapshot(event.case_value), snapshot(event.verb_meaning))
        return dict(version=13, pending=copy.deepcopy(self.pending), pending_step=self.pending_step, candidate_counter=self.candidate_counter, field_width=self.field_width,
            transient_ids=snapshot(self.transient_ids), transient_used=snapshot(self.transient_used),
            closes=tuple(self.closes.items()),
            accesses=tuple((key, tuple(value.items())) for key, value in self.accesses.items()),
            entries=tuple(dict(key=key, occurrences=entry.occurrences,
            kind=entry.kind, count=entry.count, when=entry.when,
            native=entry.native, pending=entry.pending, symbol=entry.symbol, transient=entry.transient, where=snapshot(entry.where),
            order=entry.order, support=dict(entry.support), mass=entry.mass,
            readings=entry.readings, refinement=copy.deepcopy(entry.refinement),
            timelines=tuple(tuple(map(event_state, timeline.events))
                            for timeline in entry.timelines)) for key, entry in self.entries.items()),
            names=tuple((mention, name.token, name.reading) for mention, name in self.names.items()))

    def restore(self, state, *, device=None):
        """Validate a complete pool before replacing its current context."""
        if state is None:
            self.entries, self.names, self.pending = {}, {}, {}
            self.pending_step = 0
            self.closes, self.accesses = {}, {}
            self.transient_used.zero_()
            self.candidate_counter = 0
            return
        if not isinstance(state, dict) or state.get('version') not in (8, 9, 10, 11, 12, 13):
            raise ValueError('MTM checkpoint requires transient pool version 8, 9, 10, 11 or 12')
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
            collection = isinstance(key, tuple) and len(key) in (3, 4) and key[0] == 'collection' and sentence(key[2])
            candidate = (isinstance(key, tuple) and len(key) == 2 and key[0] == 'candidate'
                         and type(key[1]) is int and 0 <= key[1] < counter)
            if (key in entries or (candidate and uses) or (not candidate and
                    (not (address(key) or named(key) or collection) or not all(map(sentence if collection else address, uses))
                     or (not named(key) and not collection and key not in uses)
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
                if state['version'] < 12:
                    if not isinstance(timeline, (tuple, list)) or len(timeline) != 2:
                        raise ValueError('invalid identity checkpoint timeline')
                    history, _retired_value = timeline
                    history = tuple((*event[:7], *event[8:]) for event in history)
                else:
                    history = timeline
                if not isinstance(history, (tuple, list)) or any(
                        not isinstance(event, (tuple, list)) or len(event) not in (10, 11) for event in history):
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
                    snapshots = [value for value in (event.before, event.case_value, event.verb_meaning,
                                 *event.modifiers) if value is not None]
                    if snapshots and any(self._value(value).shape != snapshots[0].shape for value in snapshots):
                        raise ValueError('invalid identity event snapshot')
                def move(value):
                    value = self._value(value)
                    return value if device is None else value.to(device)
                events = tuple(replace(event,
                    before=None if event.before is None else move(event.before),
                    case_value=None if event.case_value is None else move(event.case_value),
                    verb_meaning=None if event.verb_meaning is None else move(event.verb_meaning),
                    modifiers=tuple(move(item) for item in event.modifiers),
                    basis=move(event.basis) if torch.is_tensor(event.basis) else event.basis)
                    for event in events)
                if (any(not sentence(event.occurrence) or event.role not in (0, 2) or
                        type(event.verb) is not int or event.verb <= 0 or type(event.when) is not int
                        for event in events) or len({(event.occurrence, event.role) for event in events}) != len(events)):
                    raise ValueError('invalid identity checkpoint event')
                timelines.append(Timeline(events))
            refinement = record.get('refinement', {})
            if not isinstance(refinement, dict):
                raise ValueError('invalid identity checkpoint refinement')
            for member in refinement.get('members', ()):
                if (not isinstance(member, (tuple, list)) or len(member) != 2 or member[0] not in uses
                        or not (symbol(member[1]) or isinstance(member[1], tuple) and len(member[1]) == 2
                            and member[1][0] == 'bp' and type(member[1][1]) is int and member[1][1] >= 0)):
                    raise ValueError('invalid identity checkpoint member reference')
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
                for key, entry in entries.items() if key[:1] != ('collection',)) + sum(named(mention) for mention in names)):
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
                    or any(key not in entries or not value or any(
                        (stream is not None and (type(stream) is not int or stream < 0)) or
                        type(clock) is not int or not 0 <= clock <= clocks.get(stream, 0)+1
                        for stream, clock in value.items()) for key, value in accesses.items())):
                raise ValueError('invalid identity checkpoint expiry metadata')
        claimed = [entry.transient for entry in entries.values() if entry.symbol is None
                   and 'promoted_percept' not in entry.refinement]
        if (any(type(identity) is not int or identity not in self.transient_ids.tolist() for identity in claimed)
                or len(claimed) != len(set(claimed))
                or any(entry.transient is not None for entry in entries.values() if entry.symbol is not None)
                or not torch.equal(used.cpu(), torch.tensor([int(identity) in claimed for identity in self.transient_ids]))):
            raise ValueError('identity checkpoint transient assignments are inconsistent')
        self.transient_used.copy_(used.cpu())
        self.entries, self.names = entries, names
        self.pending = copy.deepcopy(state.get('pending', {}))
        self.pending_step = int(state.get('pending_step', 0))
        self.closes, self.accesses = clocks, accesses
        self.candidate_counter = counter
        self.migration_report = dict(source=state['version'], target=12,
            dropped_values=sum(len(record['timelines']) for record in state['entries']) if state['version'] < 12 else 0,
            dropped_event_after=sum(len(timeline[0]) for record in state['entries'] for timeline in record['timelines']) if state['version'] < 12 else 0)


class IdeaRows(ClauseRows, EntryRead, MemoryStore, Layer):
    """The common idea read; durability and working syntax have separate owners."""
    columns = ('slots', 'episodic_concepts', 'episodic_values', 'trust', 'referent', 'activity')
    caches = ()
    auxiliary = {'count': 'occupied seats', 'model_clock': 'next occurrence time',
        'percept_layout': 'construction geometry', 'truth_schema': 'checkpoint layout',
        'forget_passes': 'pass count', 'forget_sentences': 'kept sentence count',
        'forget_last_sentence': 'last scheduled pass', 'forget_pending': 'scheduled pass'}

    def __init__(self, width, capacity, *, percept_width, episodic_concepts=0,
                 relation_capacity=8192, read_capacity=None, content_width=None):
        super().__init__(width, width)
        self.nDim, self.capacity = int(width), int(capacity)
        self.percept_width, self.episodic_capacity = int(percept_width), int(episodic_concepts)
        if self.capacity <= 0 or not 0 <= self.percept_width <= self.nDim or self.episodic_capacity < 0:
            raise ValueError('invalid idea construction')
        self.content_width = self.nDim if content_width is None else int(content_width)
        self.when_start = self.percept_width-4 if self.percept_width >= 8 else None
        self.ltm_window = int(TheXMLConfig.get('architecture.ltmWindow', default=1024) if read_capacity is None else read_capacity)
        if self.ltm_window <= 0:
            raise ValueError('ltmWindow must be positive')
        self.register_buffer('slots', torch.zeros(self.capacity,self.nDim))
        self.register_buffer('episodic_concepts', torch.full((self.capacity,self.episodic_capacity),-1,dtype=torch.long))
        self.register_buffer('episodic_values', torch.zeros(self.capacity,self.episodic_capacity))
        self.register_buffer('trust', torch.zeros(self.capacity))
        self.register_buffer('referent', torch.stack((torch.zeros(self.capacity,dtype=torch.long),
            torch.full((self.capacity,),-1,dtype=torch.long)), -1))
        self.register_buffer('activity', torch.zeros(self.capacity))
        self.register_buffer('count', torch.tensor(0,dtype=torch.long))
        self.register_buffer('model_clock', torch.tensor(0,dtype=torch.long))
        self.register_buffer('percept_layout', torch.tensor(self.percept_width,dtype=torch.long))
        self.register_buffer('truth_schema', torch.tensor(17,dtype=torch.long))
        for name in ('forget_passes','forget_sentences','forget_last_sentence'):
            self.register_buffer(name,torch.tensor(0,dtype=torch.long))
        self.register_buffer('forget_pending',torch.tensor(False))
        from MemoryActivity import Activity
        self.activity_config = Activity.from_xml()
        import weakref
        object.__setattr__(self,'parts',LTM1(relation_capacity))
        self.parts._occurrence_store = weakref.ref(self)

    def __len__(self):
        return int(self.count)

    def clear_caches(self):
        pass

    def rebuild_caches(self):
        pass

    def validate_referents(self, state, prefix, names):
        count = int(state[prefix+'count'])
        refs = state[prefix+'referent']
        if not 0 <= count <= self.capacity or refs.shape != self.referent.shape:
            raise ValueError('invalid idea referent checkpoint column')
        kinds, values = refs[:count].unbind(-1)
        valid = ((values == -1) & (kinds == 0)) | ((kinds == SYMBOL) & (values > 0))
        valid |= (kinds == OCCURRENCE) & torch.isin(values, names[:count].to(values))
        if not bool(valid.all()):
            raise ValueError('idea referent names an unavailable occurrence or concept')

    @property
    def time(self):
        return self.slots[:,:0] if self.when_start is None else self.slots[:,self.when_start:self.when_start+4]

    def owner(self):
        ref = self.__dict__.get('_concept_owner')
        return None if ref is None else ref()

    def referent_of(self, index):
        tag, value = self.referent[int(index)].tolist()
        return ('sym',value) if tag == SYMBOL else ('ltm',self.address_domain,value) if tag == OCCURRENCE else None

    def order_of_row(self, row, seen=None):
        reference = self.referent_of(row)
        owner = self.owner()
        if reference is None and owner is not None:
            pending = owner.mtm.pending.get(self.occurrence_of(row))
            if pending is not None:
                meaning = pending[1]
                marks = dict(meaning.bindings)
                reference = marks.get('_identity') or meaning.role_refs[0]
                forward = marks.get('_forward_references', ())
                if reference is None and len(forward) == 1 and forward[0][0] == 0:
                    reference = ('sym', forward[0][1])
        if reference is None:
            return 0
        seen = set() if seen is None else set(seen)
        name = ('occurrence', self.address_domain, self.row_name(row))
        if name in seen:
            return -1
        seen.add(name)
        if reference[0] == 'ltm':
            target = self.location(reference)
            return -1 if target is None else self.order_of_row(target,seen)
        return 0 if owner is None else owner._concept_allocator.order_of(reference[1], seen)

    def occurrence_of(self, index):
        if isinstance(index,tuple) and index[:1] == ('sym',):
            return index
        if not 0 <= int(index) < len(self):
            raise IndexError('unavailable idea')
        return ('ltm',self.address_domain,self.row_name(index))

    def raw_meaning_of(self, index):
        if isinstance(index,tuple):
            return self.part_row(index[1])['meaning']
        if not 0 <= int(index) < len(self):
            raise IndexError('unavailable idea')
        meaning = replace(ConceptualMeaning.from_description(self.slots[int(index)].detach()),
            role_refs=(self.referent_of(index),None,None),sentence_kind='idea')
        return meaning

    def meaning_of(self, index):
        value = self.raw_meaning_of(index)
        return self.read_reference_bindings(index,value)

    def evidence_of(self, index):
        value = float(self.trust[int(index)])
        return max(value,0.),max(-value,0.)

    def row(self,index):
        if isinstance(index,tuple):
            return self.part_row(index[1])
        return dict(point=self.slots[int(index)],trust=float(self.trust[int(index)]),
            referent=self.referent_of(index),activity=float(self.activity[int(index)]),
            evidence=self.evidence_of(index),order=self.order_of_row(index),
            row_id=self.row_name(index),time=self.entry_time(index).clone(),
            occurrence=self.occurrence_of(index),meaning=self.meaning_of(index))

    def ideas(self):
        return torch.arange(len(self),device=self.slots.device)

    def _fit(self, value):
        if value is None:
            return self.slots.new_zeros(self.nDim)
        value=value.reshape(-1).to(self.slots)
        if len(value) == self.nDim:
            return value
        out=self.slots.new_zeros(self.nDim)
        out[:min(len(value),self.nDim)]=value[:self.nDim]
        return out

    def _validate_local_references(self, meaning):
        """Check retained operands before a write; no metadata is stored here."""
        pending = [meaning.role_refs, meaning.bindings, meaning.scope]
        while pending:
            value = pending.pop()
            if isinstance(value, dict):
                pending.extend(value.values())
            elif isinstance(value, (tuple, list)):
                if value and isinstance(value[0], str) and value[0] in ('ltm', 'thought'):
                    if (len(value) != 3 or not isinstance(value[1], str)
                            or type(value[2]) is not int or value[2] < 0):
                        raise ValueError('invalid semantic occurrence reference')
                    if value[:2] == ('ltm', self.address_domain) and self.location(tuple(value)) is None:
                        reader = self.__dict__.get('_external_memory')
                        external = None if reader is None else reader()
                        if external is None or external is self or external.location(tuple(value)) is None:
                            raise ValueError('semantic constituent occurrence is unavailable')
                else:
                    pending.extend(value)

    def _reference_value(self,reference):
        if reference is None:
            return 0,-1
        if reference[0] == 'sym' and len(reference) == 2 and int(reference[1]) > 0:
            return SYMBOL,int(reference[1])
        if reference[:2] == ('ltm',self.address_domain) and len(reference) == 3 and int(reference[2]) >= 0:
            if self.location(reference) is None:
                raise ValueError('referent names an unavailable idea')
            return OCCURRENCE,int(reference[2])
        raise ValueError('a referent must name a concept or a retained idea')

    def append_meaning(self,meaning,*,trust=0.,evidence=None,surprise=-1.,referent=None,entry=None):
        if not isinstance(meaning,ConceptualMeaning) or meaning.roles.shape[-1] != self.nDim:
            raise ValueError('idea width differs from its store')
        occupied=meaning.role_mask.nonzero().flatten()
        if len(occupied) != 1:
            raise ValueError('LTM0 stores one idea; write semantic pairs through write_clause')
        self._validate_local_references(meaning)
        if evidence is None:
            if not math.isfinite(float(trust)):
                raise ValueError('idea trust must be finite')
            trust = max(-1., min(1., float(trust)))
        pair=(max(float(trust),0.),max(-float(trust),0.)) if evidence is None else tuple(map(float,evidence))
        if len(pair) != 2 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in pair):
            raise ValueError('idea trust must be finite and in [-1, 1]')
        values=([pair[0]] if pair[0] else [])+([-pair[1]] if pair[1] else []) or [0.]
        if len(self)+len(values)>self.capacity:
            raise OverflowError('LTM0 construction capacity exhausted; forgetting is required')
        if referent is None:
            referent=meaning.role_refs[int(occupied[0])] or dict(meaning.bindings).get('_identity')
        if isinstance(referent, tuple) and referent[:1] == ('name',):
            owner = self.owner()
            if owner is None or owner.mtm.token_of(referent) is None:
                raise ValueError('pending referent names an unavailable MTM identity')
            meaning = replace(meaning, bindings={**dict(meaning.bindings), '_identity': referent})
            referent = None
        reference=self._reference_value(referent)
        if entry is None and isinstance(self,LTM0) and meaning.extension:
            from EpisodicMemory import for_meaning
            entry=for_meaning(self,meaning)
        first=len(self)
        for value in values:
            row=len(self)
            self._write_point(row,meaning.roles[int(occupied[0])])
            self.trust[row]=value
            self.referent[row]=self.referent.new_tensor(reference)
            self.count.fill_(row+1)
            if entry is not None:
                for name,data in zip(('episodic_concepts','episodic_values'),entry):
                    getattr(self,name)[row].copy_(data.detach())
            from MemoryActivity import written
            written(self,row,surprise)
            self._keep_working(row,meaning)
        from MemoryActivity import written_complete
        written_complete(self, range(first,len(self)))
        return first

    def append_idea(self,point,**kwargs):
        return self.append_meaning(ConceptualMeaning.from_description(self._fit(point)),**kwargs)

    def _keep_working(self,row,meaning):
        owner=self.owner()
        if owner is not None:
            owner.mtm.hold_pending(self.occurrence_of(row),meaning)

    def append_relation(self,part,whole,*,truth_plus=0.,truth_minus=0.,surprise=-1.):
        return ('sym',self.parts.add(part,whole,truth_plus=truth_plus,truth_minus=truth_minus,surprise=surprise))

    @torch.no_grad()
    def compact_rows(self,keep):
        keep=torch.as_tensor(keep,device=self.slots.device,dtype=torch.long)
        if len(keep.unique()) != len(keep) or bool(((keep<0)|(keep>=len(self))).any()):
            raise ValueError('invalid retained idea seats')
        keep=keep.sort().values
        kept=set(keep.tolist())
        removed=[self.occurrence_of(row) for row in range(len(self)) if row not in kept]
        for name in self.columns:
            value=getattr(self,name)
            retained=value[keep].clone()
            value.zero_()
            if name == 'episodic_concepts': value.fill_(-1)
            if name == 'referent': value[:,1]=-1
            value[:len(keep)].copy_(retained)
        self.count.fill_(len(keep))
        self.clear_caches()
        owner=self.owner()
        if owner is not None:
            for reference in removed:
                owner.mtm.pending.pop(reference,None)
                owner.mtm.forget(reference)

    @torch.no_grad()
    def reset(self):
        self.compact_rows(())
        for name in ('forget_passes','forget_sentences','forget_last_sentence','forget_pending'):
            getattr(self,name).zero_()



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



class LTM0(IdeaRows):
    """One immutable signed claim, typed referent and flat extension per time."""
    address_domain='time-v1'

    def __init__(self,nDim,capacity=1024,content_width=None,*,percept_width=None,
                 relation_capacity=8192,read_capacity=None,episodic_concepts=0):
        super().__init__(nDim,capacity,content_width=content_width,
            percept_width=nDim if percept_width is None else percept_width,
            relation_capacity=relation_capacity,read_capacity=read_capacity,
            episodic_concepts=episodic_concepts)
        if self.when_start is None:
            raise ValueError('LTM0 requires the native percept time band')

    def row_name(self,row):
        from RowTime import encoder
        return int(encoder().decode_index(self.entry_time(int(row))))

    def index_of_row(self,reference):
        # Time order is the index. Compaction keeps it; no durable duplicate
        # time/name column or concept-to-row posting is needed.
        reference=int(reference)
        lo,hi=0,len(self)
        while lo<hi:
            mid=(lo+hi)//2
            if self.row_name(mid)<reference: lo=mid+1
            else: hi=mid
        return lo if lo<len(self) and self.row_name(lo)==reference else None

    @torch.no_grad()
    def _write_point(self,row,point):
        from RowTime import next_time,write_time
        self.slots[row].copy_(point)
        write_time(self,row,next_time(self))

    def raw_meaning_of(self,index):
        value=super().raw_meaning_of(index)
        owner=self.owner()
        return value if isinstance(index,tuple) or owner is None else owner.mtm.pending_meaning(self.occurrence_of(index),value)

    def validate_checkpoint(self, state, prefix):
        """Reject unrepresentable legacy rows before importing any store."""
        from RowTime import checkpoint_times
        version=int(state.get(prefix+'truth_schema',-1))
        if version not in (16,17):
            raise ValueError('unsupported LTM0 checkpoint layout')
        if state[prefix+'slots'].shape != self.slots.shape:
            raise ValueError('LTM0 checkpoint capacity/width differs from construction')
        count=int(state[prefix+'count'])
        ticks=checkpoint_times(state[prefix+'slots'],count,int(state[prefix+'percept_layout']))
        if version == 17:
            self.validate_referents(state, prefix, ticks)
        if version == 16:
            plus,minus=state[prefix+'truth_plus'],state[prefix+'truth_minus']
            if (plus.shape != self.trust.shape or minus.shape != plus.shape
                    or plus.dtype != self.trust.dtype or minus.dtype != plus.dtype
                    or not bool(torch.isfinite(plus).all() & torch.isfinite(minus).all())
                    or bool(((plus < 0) | (plus > 1) | (minus < 0) | (minus > 1)).any())):
                raise ValueError('invalid legacy idea evidence')
            if bool((plus[:count].ne(0) & minus[:count].ne(0)).any()):
                raise ValueError('legacy idea has both truth lanes; checkpoint cannot preserve them in one signed row')
        return ticks

    def prepare_checkpoint(self,state,prefix):
        import warnings
        ticks=self.validate_checkpoint(state,prefix)
        version=int(state[prefix+'truth_schema'])
        count=int(state[prefix+'count'])
        order=ticks.argsort(stable=True)
        if version == 16:
            addresses=state.get(prefix+'addressing.address_keys')
            self._legacy_times={} if addresses is None else dict(zip(addresses[:count].tolist(),ticks.tolist()))
            plus,minus=state[prefix+'truth_plus'],state[prefix+'truth_minus']
            state[prefix+'trust']=torch.where(plus > 0,plus,-minus)
            state[prefix+'referent']=self.referent.clone()
            state[prefix+'activity']=torch.where(torch.arange(self.capacity,device=self.activity.device)<count,
                self.activity_config.default,0.).to(self.activity)
            for name in ('episodic_concepts','episodic_values'):
                if state[prefix+name].ndim != 3:
                    raise ValueError('legacy entry has no order axis')
            ids=state[prefix+'episodic_concepts'].flatten(1)
            values=state[prefix+'episodic_values'].flatten(1)
            ranking=values.abs().masked_fill(ids<0,-1).argsort(dim=-1,descending=True,stable=True)[:,:self.episodic_capacity]
            picked_ids,picked_values=ids.gather(1,ranking),values.gather(1,ranking)
            padding=self.episodic_capacity-picked_ids.shape[1]
            state[prefix+'episodic_concepts']=torch.nn.functional.pad(picked_ids,(0,padding),value=-1)
            state[prefix+'episodic_values']=torch.nn.functional.pad(picked_values,(0,padding))
        allowed={*self.columns,*self.auxiliary}
        for key in tuple(state):
            if key.startswith(prefix) and key[len(prefix):] not in allowed:
                state.pop(key)
                warnings.warn('LTM0 checkpoint: dropped retired field or cache '+key[len(prefix):])
        state[prefix+'truth_schema']=self.truth_schema.clone()
        state[prefix+'model_clock']=self.model_clock.new_tensor(max(int(state[prefix+'model_clock']),int(ticks.max())+1 if count else 0))
        for name in self.columns:
            value,target=state.get(prefix+name),getattr(self,name)
            if value is None or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('LTM0 checkpoint column differs from construction: '+name)
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError('nonfinite idea checkpoint column: '+name)
            if count:
                value=value.clone();value[:count]=value[:count][order]
                state[prefix+name]=value
        if bool((state[prefix+'trust'].abs()>1).any()) or bool(((state[prefix+'activity']<0)|(state[prefix+'activity']>1)).any()):
            raise ValueError('invalid idea trust or activity')
    def _load_from_state_dict(self,state,prefix,*args,**kwargs):
        self.prepare_checkpoint(state,prefix)
        super()._load_from_state_dict(state,prefix,*args,**kwargs)




class LTM1(MemoryStore, nn.Module):
    """A tagged ordered pair, its two trust lanes, and its activity."""
    OCCURRENCE, SYMBOL, RELATION = OCCURRENCE, SYMBOL, RELATION
    columns = ('refs', 'ref_kinds', 'truth_plus', 'truth_minus', 'activity')
    caches = ('_pairs', '_parts', '_wholes', '_references', '_index_version',
              'relation_order', '_order_revision', 'effective_plus',
              'effective_minus', '_effective_source')
    auxiliary = {'count': 'occupied relation rows', 'layout_version': 'checkpoint schema'}

    def __init__(self, capacity=8192):
        super().__init__()
        self.capacity = int(capacity)
        if self.capacity < 2:
            raise ValueError('LTM1 needs a reserved zero and a relation seat')
        self.allocate = self.handle_reserved = None
        self.register_buffer('refs', torch.full((self.capacity, 2), -1, dtype=torch.long))
        self.register_buffer('ref_kinds', torch.zeros(self.capacity, 2, dtype=torch.int8))
        for name in ('truth_plus', 'truth_minus', 'activity'):
            self.register_buffer(name, torch.zeros(self.capacity))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))
        self.register_buffer('layout_version', torch.tensor(VERSION, dtype=torch.long))
        self.register_buffer('relation_order', torch.full((self.capacity,), -1, dtype=torch.long), persistent=False)
        for name in ('effective_plus', 'effective_minus'):
            self.register_buffer(name, torch.zeros(self.capacity), persistent=False)
        from MemoryActivity import Activity
        self.activity_config = Activity.from_xml()
        self._concept_owner = None
        self._symbol_epoch = 0
        self.clear_caches()

    def clear_caches(self):
        self._pairs, self._parts, self._wholes, self._references = {}, {}, {}, {}
        self._index_version = self._order_revision = None
        self.relation_order.fill_(-1)
        self._effective_source = None
        self.effective_plus.zero_()
        self.effective_minus.zero_()

    def rebuild_caches(self):
        self._index()
        self._sort()
        self.refresh_effective()

    def _reserved_handle(self, row):
        if self.handle_reserved is not None and self.handle_reserved(row):
            return True
        if self.rows_referencing(SYMBOL, row):
            return True
        ref = getattr(self, '_occurrence_store', None)
        store = None if ref is None else ref()
        return store is not None and bool(store.episodic_concepts[:len(store)].eq(row).any())

    def __len__(self):
        return int(self.count)

    @property
    def revision(self):
        return (id(self.refs), self.refs._version, self.ref_kinds._version,
                self.truth_plus._version, self.truth_minus._version, len(self))

    @property
    def _topology_revision(self):
        return (id(self.refs), self.refs._version, self.ref_kinds._version, len(self))

    @property
    def symbol_revision(self):
        return (id(self), self._symbol_epoch)

    @torch.no_grad()
    def refresh_effective(self):
        """Read endpoint confidence in dependency order, without evidential access.

        The source pair is unchanged. An allocated concept contributes its
        for-lane radius, and a relation endpoint its stronger effective lane.
        Operators contribute no factor.
        """
        owner = None if self._concept_owner is None else self._concept_owner()
        trust = None if owner is None else owner.similarity_codebook.trust
        revision = (self.revision, None if trust is None else trust.revision)
        if revision == self._effective_source:
            return
        self._sort()
        from ConceptTrust import positive_length
        def endpoint(kind, value):
            if kind == RELATION:
                return max(float(self.effective_plus[value]), float(self.effective_minus[value]))
            if kind == SYMBOL and owner is not None and value in owner._concept_allocator.placement:
                return positive_length(owner, value)
            return 1.
        for row in self.relation_order[:len(self)].tolist():
            scale = min(endpoint(*ref) for ref in self.references(row))
            self.effective_plus[row] = self.truth_plus[row] * scale
            self.effective_minus[row] = self.truth_minus[row] * scale
        self._effective_source = revision

    @property
    def effective_revision(self):
        self.refresh_effective()
        # Removing the last relation performs no effective-pair writes. A
        # derived reader must still observe that the underlying table changed.
        return (self.revision, self.effective_plus._version, self.effective_minus._version)

    def evidence(self, row):
        if not self.contains(row):
            raise IndexError('unavailable relation')
        self.refresh_effective()
        from MemoryActivity import accessed
        accessed(self, row)
        return self.effective_plus[row], self.effective_minus[row]

    @staticmethod
    def reference(value, kind=SYMBOL):
        if isinstance(value, (tuple, list)):
            if len(value) != 2:
                raise ValueError('a relation reference requires its tag and value')
            kind, value = value
        kind, value = int(kind), int(value)
        if kind not in (OCCURRENCE, SYMBOL, RELATION) or value == -1 or (kind != OCCURRENCE and value <= 0):
            raise ValueError('invalid tagged relation reference')
        return kind, value

    def _index(self):
        if self._index_version == self._topology_revision:
            return
        self._pairs, self._parts, self._wholes, self._references = {}, {}, {}, {}
        for row in self.indices().tolist():
            left, right = self.references(row)
            if (left, right) in self._pairs:
                raise ValueError('duplicate tagged relation')
            self._index_pair(row, left, right)
        self._index_version = self._topology_revision

    def _index_pair(self, row, left, right):
        import bisect
        self._pairs[left, right] = row
        self._parts.setdefault(right, []).append(left)
        self._parts[right].sort(key=lambda ref: self._pairs[ref, right])
        self._wholes.setdefault(left, []).append(right)
        self._wholes[left].sort(key=lambda ref: self._pairs[left, ref])
        for reference in {left, right}:
            bisect.insort(self._references.setdefault(reference, []), row)

    def protected_references(self, rows=None):
        pending = list(self.indices().tolist() if rows is None else rows)
        seen, references = set(), set()
        while pending:
            row = pending.pop()
            if row in seen:
                continue
            seen.add(row)
            for reference in self.references(row):
                references.add(reference)
                if reference[0] == RELATION:
                    if not self.contains(reference[1]):
                        raise ValueError('dangling relation reference')
                    pending.append(reference[1])
        return references

    @torch.no_grad()
    def add(self, part, whole, *, ref_kinds=(SYMBOL, SYMBOL), truth_plus=0., truth_minus=0.,
            surprise=-1., symbol=None):
        left, right = self.reference(part, ref_kinds[0]), self.reference(whole, ref_kinds[1])
        if (SYMBOL, NULL_CONCEPT) in (left, right):
            raise ValueError('a null-concept cannot be a relation endpoint')
        if any(not math.isfinite(float(x)) or not 0 <= x <= 1 for x in (truth_plus, truth_minus)):
            raise ValueError('relation trust has two independent lanes in [0, 1]')
        if not math.isfinite(float(surprise)) or not (surprise == -1. or 0 <= surprise <= 1.):
            raise ValueError('relation surprise must be unknown or in [0, 1]')
        for kind, value in (left, right):
            if kind == RELATION and not self.contains(value):
                raise ValueError('relation child must exist before its parent')
        row = self.find(left, right)
        self._sort()
        new = row is None
        if new:
            if symbol is not None:
                row = int(symbol)
            elif self.allocate is None:
                free = self.ref_kinds[:, 0].eq(0).nonzero().flatten().tolist()
                row = next((cid for cid in free if cid > 0 and not self._reserved_handle(cid)), None)
                if row is None:
                    raise OverflowError('LTM1 capacity exhausted; forgetting is required')
            else:
                row = int(self.allocate())
            if not 0 < row < self.capacity:
                raise OverflowError('LTM1 handle exceeds its constructed capacity')
            if self.contains(row):
                raise RuntimeError('a live relation cannot be reassigned')
            self.refs[row] = self.refs.new_tensor((left[1], right[1]))
            self.ref_kinds[row] = self.ref_kinds.new_tensor((left[0], right[0]))
            self.relation_order[len(self)] = row
            self.count += 1
            self._order_revision = self._topology_revision
        elif symbol is not None and int(symbol) != row:
            raise ValueError('a tagged pair already has a different handle')
        positive, negative = max(float(self.truth_plus[row]), float(truth_plus)), max(float(self.truth_minus[row]), float(truth_minus))
        if right[0] == SYMBOL and (left[0] == SYMBOL or right[1] in getattr(self, '_class_rows', ())):
            if new or positive != float(self.truth_plus[row]) or negative != float(self.truth_minus[row]):
                self._symbol_epoch += 1
        self.truth_plus[row], self.truth_minus[row] = positive, negative
        from MemoryActivity import written, accessed
        if new:
            written(self, row, surprise)
            self._index_pair(row, left, right)
            self._index_version = self._topology_revision
        else:
            accessed(self, row)
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
            getattr(self, name)[erase] = -1 if name == 'refs' else 0
        self.count.fill_(len(indices))
        self._index()
        self._sort()

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        import warnings
        version = state.get(prefix+'layout_version')
        if version is None or int(version) not in (5, VERSION):
            raise ValueError('unsupported LTM1 checkpoint layout')
        allowed = {*self.columns, 'count', 'layout_version'}
        for key in tuple(state):
            if key.startswith(prefix) and key[len(prefix):] not in allowed:
                state.pop(key)
                warnings.warn('LTM1 checkpoint: dropped retired field or cache '+key[len(prefix):])
        state[prefix+'layout_version'] = self.layout_version.clone()
        if prefix+'activity' not in state:
            state[prefix+'activity'] = torch.where(state[prefix+'ref_kinds'][:,0] != 0,
                torch.full_like(self.activity,self.activity_config.default), 0.)
        for name in self.columns:
            value = state.get(prefix+name)
            target = getattr(self, name)
            if value is None or value.shape != target.shape or value.dtype != target.dtype:
                raise ValueError('relation checkpoint column differs from construction')
        refs, kinds = state[prefix+'refs'], state[prefix+'ref_kinds']
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
        if int(occupied.sum()) != int(state.get(prefix+'count', -1)):
            raise ValueError('relation count disagrees with occupied rows')
        self.ordered_rows(refs,kinds)
        for name in ('truth_plus','truth_minus','activity'):
            values = state[prefix+name]
            if not bool(torch.isfinite(values).all() & (values >= 0).all() & (values <= 1).all()):
                raise ValueError('invalid restored relation '+name)
        super()._load_from_state_dict(state, prefix, *args, **kwargs)
        self.clear_caches()
        self._symbol_epoch += 1
        self.rebuild_caches()

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


    def references(self, row):
        if not self.contains(row):
            raise IndexError('unavailable relation')
        return tuple(zip(self.ref_kinds[int(row)].tolist(), self.refs[int(row)].tolist()))


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



class STM1(LTM1):
    """Ordered working branches, with LTM1's storage and tagged-reference reads."""
    def expire(self):
        self.retain(())


class STM0(IdeaRows):
    """Transient ideas and syntax, without LTM0's durable time contract."""
    address_domain='stm-v1'
    columns=(*IdeaRows.columns,'row_names')

    def __init__(self,width,capacity,*,percept_width=0,relation_capacity=2):
        super().__init__(width,capacity,percept_width=percept_width,
            relation_capacity=relation_capacity,read_capacity=capacity)
        self.register_buffer('row_names',torch.full((capacity,),-1,dtype=torch.long))
        import weakref
        object.__setattr__(self,'parts',STM1(relation_capacity))
        self.parts._occurrence_store=weakref.ref(self)
        self.working={}
        self.generation=0

    def row_name(self,row):
        return int(self.row_names[int(row)])

    def index_of_row(self,reference):
        found=self.row_names[:len(self)].eq(int(reference)).nonzero().flatten()
        return int(found[0]) if len(found) else None

    def _write_point(self,row,point):
        # Functional replacement retains the live graph of the pinned tree.
        with torch.enable_grad():
            self.slots=self.slots.index_copy(0,self.count.new_tensor([row]),point[None])
        name = self.__dict__.get('_next_row_name', int(self.model_clock))
        self.row_names[row] = name
        self.model_clock.fill_(max(int(self.model_clock), name + 1))
        if '_next_row_name' in self.__dict__:
            self._next_row_name = name + 1

    def _keep_working(self,row,meaning):
        super()._keep_working(row,meaning)
        reference=self.referent_of(row)
        self.working[self.row_name(row)]=replace(meaning,
            role_refs=(reference,*meaning.role_refs[1:]),
            sentence_kind=meaning.sentence_kind or 'idea').metadata()

    def raw_meaning_of(self,index):
        if isinstance(index,tuple):
            return self.part_row(index[1])['meaning']
        value=replace(super().raw_meaning_of(index),
            **self.working.get(self.row_name(index),{}))
        owner=self.owner()
        return value if owner is None else owner.mtm.pending_meaning(self.occurrence_of(index),value)

    def append_meaning(self,meaning,*,name=None,**kwargs):
        if name is not None:
            if type(name) is not int or name < 0 or self.index_of_row(name) is not None:
                raise ValueError('working occurrence name must be fresh and nonnegative')
            self._next_row_name = name
        try:
            return super().append_meaning(meaning,**kwargs)
        finally:
            self.__dict__.pop('_next_row_name', None)

    def _load_from_state_dict(self, state, prefix, *args, **kwargs):
        self.validate_referents(state, prefix, state[prefix+'row_names'])
        super()._load_from_state_dict(state, prefix, *args, **kwargs)

    def compact_rows(self,keep):
        keys={self.row_name(row) for row in keep}
        super().compact_rows(keep)
        self.working={key:value for key,value in self.working.items() if key in keys}

    def expire(self):
        if len(self):
            self.slots=self.slots.detach().clone()
        self.reset()
        self.working.clear()
        self.parts.expire()
        self.generation+=1
