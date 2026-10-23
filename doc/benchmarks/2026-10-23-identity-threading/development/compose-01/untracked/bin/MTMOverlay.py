"""Differentiable identity transactions carried through the compose loop.

Locations index this transaction, never conceptual seats or durable addresses.
The owner stages frame-accessible values outside the compiled body. All live
methods below operate on tensors and the kept journal supplies commit order.
"""
import torch
from torch import nn


class MTMOverlay(nn.Module):
    # Stack columns: identity, predicate, object, resolved, word, source.
    # Journal columns: result identity, subject, object, predicate, bind, event.
    def __init__(self, binary, unary):
        super().__init__()
        def table(rules):
            return torch.tensor([(
                getattr(rule, 'head_role', 0),
                getattr(rule, 'determiner_mode', None) in ('mint', 'bind'),
                rule.method_name == 'bind',
                getattr(rule, 'clause_form', None) == 'VP',
                getattr(rule, 'clause_form', None) == 'S',
                getattr(rule, 'relation_kind', None) == 'equal',
                getattr(rule, 'relation_kind', None) in ('part', 'whole', 'equal'),
            ) for rule in rules] or [(0,) * 7], dtype=torch.long)
        self.register_buffer('binary', table(binary), persistent=False)
        self.register_buffer('unary', table(unary), persistent=False)

    @staticmethod
    def empty(like, *, journal, words):
        batch, stack, width = like.shape
        # One introduction per word or selected operation. This is a write
        # journal, not extra STM capacity or a second set of accessible frames.
        count = words + journal
        return (like.new_zeros(batch, count, width),
                torch.zeros(batch, count, dtype=torch.bool, device=like.device),
                torch.zeros(batch, stack, 6, dtype=torch.long, device=like.device),
                torch.zeros(batch, count, 2, dtype=torch.long, device=like.device),
                torch.zeros(batch, journal, 6, dtype=torch.long, device=like.device))

    @staticmethod
    def stage(resolver, frames, like, *, journal, words):
        """Host boundary: no pool access is needed during composition."""
        records = [resolver.individuals(resolver.accessible(row)) for row in frames]
        base = max(map(len, records), default=0)
        values, valid, stack, origins, actions = MTMOverlay.empty(like,
            journal=journal + base, words=words)
        actions = actions[:, :journal]
        for batch, row in enumerate(records):
            for index, (identity, record) in enumerate(row.items()):
                value = record.get('value', record['content']).to(like)
                values[batch, index, :value.numel()] = value
                valid[batch, index] = True
                origins[batch, index] = origins.new_tensor(
                    (1, identity) if isinstance(identity, int) else (2, identity[-1]))
        return values, valid, stack, origins, actions

    @staticmethod
    def read(values, handles):
        width = values.shape[-1]
        return values.gather(1, (handles - 1).clamp_min(0)[..., None].expand(*handles.shape, width))

    @staticmethod
    def write(values, handles, content, active):
        at = torch.arange(values.shape[1], device=values.device)[None]
        return torch.where((active[:, None] & (at == handles[:, None] - 1))[..., None],
                           content[:, None], values)

    @staticmethod
    def push(state, active, value, *, word, order, index, words, resolved):
        values, valid, stack, origins, journal = state
        base = values.shape[1] - words - journal.shape[1]
        handle = (base + index + 1).expand(len(value))
        # The reading stamp makes a provisional object. Its grammatical
        # participant position is established by a later kept operation.
        named = active & (order == 1)
        values = MTMOverlay.write(values, handle, value, named)
        origin = torch.stack((torch.full_like(handle, 4), index.expand_as(handle)), -1)
        origins = MTMOverlay.write(origins, handle, origin, named)
        member = torch.stack((torch.where(named, handle, 0), word, torch.zeros_like(word),
                              resolved.long(), word, torch.zeros_like(word)), -1)
        stack = torch.where(active[:, None, None],
            torch.cat((member[:, None], stack[:, :-1]), 1), stack)
        return values, valid, stack, origins, journal

    @staticmethod
    def refresh(state, value, source):
        values, valid, stack, _, _ = state
        handles = stack[..., 0].gather(1, source)
        present = valid.gather(1, (handles - 1).clamp_min(0)) & (handles > 0)
        return torch.where(present[..., None], MTMOverlay.read(values, handles), value)

    @staticmethod
    def menu(state, value, source):
        values, valid, stack, origins, _ = state
        batch, positions, width = value.shape
        handles = stack[..., 0].gather(1, source)
        columns = torch.arange(values.shape[1], device=value.device) + 1
        available = valid[:, None].expand(batch, positions, -1)
        available = available & (columns[None, None] != handles[..., None])
        # Dummy is present even when there are candidates. Locations are
        # enumerated, never pre-ranked or truncated by a second chooser.
        candidates = values[:, None].expand(batch, positions, -1, width)
        own = value[:, :, None].expand_as(candidates)
        joined = torch.maximum(own, candidates)
        selected = columns[None, None].expand(batch, positions, -1)
        refs = torch.where(origins[..., 0] != 3, origins[..., 1], -1)
        refs = torch.where(origins[..., 0] == 4, -1, refs)
        return (torch.cat((value[:, :, None], joined), 2),
                torch.cat((value[:, :, None], candidates), 2),
                torch.cat((torch.zeros_like(handles)[..., None], selected), 2),
                torch.cat((torch.zeros_like(handles)[..., None], refs[:, None].expand_as(selected)), 2),
                torch.cat((torch.ones_like(handles, dtype=torch.bool)[..., None], available), 2))

    def apply(self, state, choice, operands, *, column, words):
        values, valid, stack, origins, journal = state
        batch, size, _ = stack.shape
        binary, unary = choice.kind == 1, choice.kind == 2
        applied = choice.applied & (binary | unary)
        rule = torch.where(binary[:, None], self.binary[choice.local_op.clamp(0, len(self.binary)-1)],
                           self.unary[choice.local_op.clamp(0, len(self.unary)-1)])
        selected = stack.gather(1, choice.position.clamp(0, size-1)[:, None, None].expand(batch, 1, 6))[:, 0]
        left = torch.where(binary[:, None], stack[:, min(1, size-1)], selected)
        right = torch.where(binary[:, None], stack[:, 0], selected)
        head = torch.where((rule[:, 0] == 2)[:, None], right, left)
        parent = torch.where((unary | (rule[:, 0] > 0))[:, None], head, torch.zeros_like(head))
        bind = applied & rule[:, 2].bool()
        determined = applied & rule[:, 1].bool()
        base = values.shape[1] - words - journal.shape[1]
        fresh = base + words + column + 1
        fresh = fresh.expand(batch)
        selected_identity = choice.identity if choice.identity is not None else torch.zeros_like(fresh)
        handle = torch.where(bind & (selected_identity > 0), selected_identity,
            torch.where(bind & (left[:, 0] > 0), left[:, 0], fresh))
        opening = bind | determined
        values = self.write(values, handle, choice.candidate, opening)
        positions = torch.arange(values.shape[1], device=values.device)[None]
        valid = valid | (opening[:, None] & (positions == handle[:, None]-1))
        origins = self.write(origins, fresh,
            torch.stack((torch.full_like(fresh, 3), column.expand(batch)), -1), opening & (handle == fresh))
        parent = torch.where(opening[:, None], torch.stack((handle, head[:, 1], head[:, 2],
            torch.where(bind & (selected_identity > 0), 1, head[:, 3]), head[:, 4], column.expand(batch)+1), -1), parent)
        # A participant without an explicit bind remains its own object.
        for member, content, participant in (
                (left, operands[:, 0], rule[:, 4].bool() | rule[:, 6].bool()),
                (right, operands[:, 1], rule[:, 3].bool() | rule[:, 6].bool())):
            h = member[:, 0]
            new = applied & participant & (h > 0) & ~valid.gather(1, (h-1).clamp_min(0)[:, None])[:, 0]
            values = self.write(values, h, content, new)
            valid |= new[:, None] & (positions == h[:, None]-1)
        vp = applied & rule[:, 3].bool()
        parent = torch.where(vp[:, None], torch.stack((torch.zeros_like(fresh), left[:, 1], right[:, 0],
            left[:, 3] & right[:, 3], left[:, 4], column.expand(batch)+1), -1), parent)
        sentence = applied & (rule[:, 4].bool() | rule[:, 5].bool())
        subject, obj, verb = left[:, 0], right[:, 2], right[:, 1]
        obj = torch.where(rule[:, 5].bool(), right[:, 0], obj)
        # Equality's complement is its new state; its mutual-parthood result
        # remains the ordinary grammatical result in the operation journal.
        result = torch.where(rule[:, 5, None].bool(), operands[:, 1], choice.candidate)
        update = sentence & (subject > 0) & right[:, 3].bool()
        values = self.write(values, subject, result, update)
        parent = torch.where(sentence[:, None], torch.stack((subject, verb, obj,
            right[:, 3], left[:, 4], column.expand(batch)+1), -1), parent)
        event = torch.stack((parent[:, 0], torch.where(sentence, subject, 0),
            torch.where(sentence, obj, 0), torch.where(sentence, verb, 0),
            torch.where(bind, selected_identity, -1), update.long()), -1)
        journal = self.write(journal, column.expand(batch)+1, event, applied)
        rewritten = torch.where((unary & applied)[:, None, None] &
            (torch.arange(size, device=values.device)[None, :, None] == choice.position[:, None, None]), parent[:, None], stack)
        folded = torch.cat((parent[:, None], stack[:, 2:], torch.zeros_like(stack[:, :1])), 1) if size > 1 else stack
        stack = torch.where((binary & applied)[:, None, None], folded, rewritten)
        if size > 1:
            stack = torch.where((choice.applied & (choice.kind == 3))[:, None, None],
                torch.cat((stack[:, 1:2], stack[:, :1], stack[:, 2:]), 1), stack)
        return values, valid, stack, origins, journal
