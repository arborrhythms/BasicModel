"""Fixed-shape grammatical clause scope carried beside the numerical STM."""
import torch
from torch import nn


class ClauseScope(nn.Module):
    """Each occupied slot carries its kind and a native or trial-local reference.

    Durable references may use either int64 sign. LOCAL marks a temporary
    relative clause in this derivation's journal; the winning closing resolves
    it to a shared row. Absolute clauses retain their numerical point.
    """
    RELATIVE, GENERIC, SENTENCE, PREDICATE, LOCAL = 1, 2, 4, 8, 16

    def __init__(self, binary, unary):
        super().__init__()

        def table(rules):
            rows = []
            for rule in rules:
                form = getattr(rule, 'clause_form', None)
                relative = bool(getattr(rule, 'relation_kind', None))
                sentence = relative or form == 'S'
                predicate = form == 'VP'
                head = getattr(rule, 'head_role', 0)
                modes = dict(getattr(rule, 'reference_kinds', ()))
                generic = any(mode in ('generic', 'kind') for mode in modes.values())
                particular = any(mode in ('particular', 'name', 'pronoun')
                                 for mode in modes.values())
                rows.append((relative, sentence, predicate, head, generic, particular))
            return torch.tensor(rows or [(0,) * 6], dtype=torch.long)
        self.register_buffer('binary', table(binary), persistent=False)
        self.register_buffer('unary', table(unary), persistent=False)
        self.register_buffer('binary_same_reference', torch.tensor(
            [getattr(rule, 'same_reference_idempotent', False) for rule in binary]
            or [False]), persistent=False)

    @staticmethod
    def empty(buffer):
        result = torch.zeros((*buffer.shape[:2], 2), dtype=torch.long, device=buffer.device)
        result[..., 1] = -1
        return result

    @staticmethod
    def push(state, active, references):
        value = torch.stack((torch.zeros_like(references), references), -1)
        shifted = torch.cat((value[:, None], state[:, :-1]), 1)
        return torch.where(active[:, None, None], shifted, state)

    @classmethod
    def slots(cls, state, depth):
        occupied = torch.arange(state.shape[1], device=state.device)[None] < depth[:, None]
        relative = ((state[..., 0].bitwise_and(cls.RELATIVE) != 0) & occupied).any(-1)
        # Two slots are an unfinished subject/predicate phrase. The existing
        # one-or-three row contract cannot admit that partial form.
        return torch.where(relative & (depth != 2), 3, 1)

    @staticmethod
    def reset(state, active):
        blank = torch.zeros_like(state)
        blank[..., 1] = -1
        return torch.where(active[:, None, None], blank, state)

    @classmethod
    def publish_name(cls, stm, scope, ended):
        """A relative result occupies its slot by name, never by one operand."""
        from References import address_code
        buffer, *rest = stm
        named = ((scope[..., 0].bitwise_and(cls.RELATIVE | cls.LOCAL)
                  == cls.RELATIVE | cls.LOCAL) & (ended == 3)[:, None])
        names = address_code(scope[..., 1], buffer.shape[-1], like=buffer)
        return (torch.where(named[..., None], names, buffer), *rest)

    @classmethod
    def resolve(cls, state, choice, references, relative):
        B, K, _ = state.shape
        positions = torch.arange(K, device=state.device)[None]
        updated = state
        for role in range(2):
            position = torch.where(choice.kind == 1, torch.full_like(
                choice.position, 1-role), choice.position)
            used = choice.applied & ((choice.kind == 1) | (role == 0))
            selected = used[:, None] & (positions == position[:, None])
            flags = updated[..., 0] | torch.where(
                relative[:, role, None], cls.RELATIVE | cls.SENTENCE, 0)
            ids = references[:, role, None].expand(B, K)
            # A selected durable operand replaces any old journal-local tag.
            flags = torch.where(ids == updated[..., 1], flags,
                                flags.bitwise_and(~cls.LOCAL))
            replacement = torch.stack((flags, ids), -1)
            updated = torch.where(selected[:, :, None], replacement, updated)
        return updated

    def apply(self, state, choice, slot):
        B, K, _ = state.shape
        binary = choice.kind == 1
        unary = choice.kind == 2
        table_b = self.binary.to(state.device)[choice.local_op.clamp(0, self.binary.shape[0] - 1)]
        table_u = self.unary.to(state.device)[choice.local_op.clamp(0, self.unary.shape[0] - 1)]
        rule = torch.where(binary[:, None], table_b, table_u)
        selected = state.gather(1, choice.position.clamp(
            0, K - 1)[:, None, None].expand(B, 1, 2))[:, 0]
        left = torch.where(binary[:, None], state[:, min(1, K - 1)], selected)
        right = torch.where(binary[:, None], state[:, 0], selected)
        headed = rule[:, 3] > 0
        head = torch.where((rule[:, 3] == 1)[:, None], left, right)
        inherited = torch.where(headed, head[:, 0], left[:, 0].bitwise_or(right[:, 0]))
        # A headed predicate retains the object's scoped clause even though
        # its numerical value and generic/name status come from the head.
        relative = (left[:, 0].bitwise_or(right[:, 0]).bitwise_and(self.RELATIVE) != 0) | rule[:, 0].bool()
        relative |= rule[:, 1].bool() & (left[:, 0].bitwise_and(self.GENERIC) != 0)
        predicate = rule[:, 2].bool()
        sentence = rule[:, 1].bool() | (binary & relative & ~predicate & ~headed)
        generic = (headed & (head[:, 0].bitwise_and(self.GENERIC) != 0)) | rule[:, 4].bool()
        generic &= ~rule[:, 5].bool()
        flags = relative.long() + self.GENERIC * generic.long()
        flags += self.SENTENCE * sentence.long() + self.PREDICATE * predicate.long()
        preserved = headed | unary
        flags = torch.where(preserved & ~sentence & ~predicate,
                            flags | inherited.bitwise_and(self.SENTENCE), flags)
        previous_ref = torch.where(headed, head[:, 1], selected[:, 1])
        already_ended = unary & (selected[:, 0].bitwise_and(self.SENTENCE) != 0)
        closing = sentence & ~already_ended & choice.applied
        local_ref = -2 - torch.as_tensor(slot, device=state.device).expand(B)
        reference = torch.where(sentence & relative, local_ref, -1)
        retained = preserved & ~closing
        reference = torch.where(retained, previous_ref, reference)
        previous_flags = torch.where(headed, head[:, 0], selected[:, 0])
        local = torch.where(retained, previous_flags.bitwise_and(self.LOCAL) != 0,
                            sentence & relative)
        flags = flags | torch.where(local, self.LOCAL, 0)
        parent = torch.stack((flags, reference), -1)
        same_reference = (binary & self.binary_same_reference.to(state.device)[
            choice.local_op.clamp(0, self.binary_same_reference.shape[0]-1)]
            & (left[:, 1] == right[:, 1]) & (left[:, 1] != -1) & (left[:, 1] != 0))
        parent = torch.where(same_reference[:, None], left, parent)
        closing = closing & ~same_reference
        rewritten = torch.where((unary & choice.applied)[:, None, None]
                                & (torch.arange(K, device=state.device)[None, :, None] == choice.position[:, None, None]),
                                parent[:, None], state)
        blank = torch.zeros_like(state[:, :1])
        blank[..., 1] = -1
        folded = torch.cat((parent[:, None], state[:, 2:], blank), 1) if K > 1 else state
        result = torch.where((binary & choice.applied)[:, None, None], folded, rewritten)
        return result, torch.where(closing, torch.where(relative, 3, 1), 0)
