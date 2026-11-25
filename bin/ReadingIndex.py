"""Action order and operand provenance of one recorded grammar walk."""
import torch


def recorded_program(order, active, ids, leaves, kind, op, offsets, bind, closing_width,
                     binary_count, stop, target_width, sentence, check_capacity):
    """Lay out the operations already selected by each walk in arrival order."""
    from Meaning import require_sentence_tensor
    B, W = active.shape
    S = kind.shape[1]
    columns = torch.arange(W, device=active.device)
    # Index W supplies an inactive word even for an empty page.
    def with_empty(table, fill):
        return torch.nn.functional.pad(table, (0, 1), value=fill)
    last = with_empty(torch.where(active, columns, -1), -1).amax(-1)
    sid = with_empty(ids, -1).gather(1, torch.where(last >= 0, last, W)[:, None]) if sentence is None else sentence
    in_sentence = active & (ids == sid)
    final = with_empty(torch.where(in_sentence, columns, -1), -1).amax(-1)
    valid = (order >= 0) & (order < W)
    keep = with_empty(in_sentence & leaves, False).gather(1, torch.where(valid, order, W))
    # Stable partition keeps the actual fixation order, including repeated words.
    word_sort = (~keep).long().argsort(dim=1, stable=True)
    words = torch.where(keep, order, -1).gather(1, word_sort)
    word = words >= 0
    slots = 3 * words[..., None] + torch.arange(3, device=active.device)
    slots = torch.where(word[..., None], slots, -1)
    shift = torch.nn.functional.pad(torch.zeros_like(slots, dtype=torch.bool), (1, 0), value=True)
    shift = (shift & word[..., None]).reshape(B, -1)
    slots = torch.nn.functional.pad(slots, (1, 0), value=-1).reshape(B, -1)
    boundary = 3 * W + torch.where(final == last, 0, (final + 1) * closing_width)
    closing = boundary[:, None] + torch.arange(closing_width, device=active.device)
    slots = torch.cat((slots, torch.where(final[:, None] >= 0, closing, -1)), 1)
    shift = torch.nn.functional.pad(shift, (0, closing_width))
    source = torch.where((slots >= 0) & (slots < S), slots, S)
    kinds = torch.where(shift, 0, with_empty(kind, -1).gather(1, source))
    local = with_empty(op, -1).gather(1, source)
    positions = with_empty(offsets, -1).gather(1, source)
    args = torch.where(shift, shift.long().cumsum(1) - 1, torch.where(kinds == 2, positions, -1))
    binary = (kinds == 1) & ~with_empty(bind, False).gather(1, source)
    delta = shift.long() - binary.long()
    depth = delta.cumsum(1) - delta
    require_sentence_tensor((~(binary | (kinds == 3)) | (depth >= 2)).all(),
                            'binary operation underflows the recorded forest')
    require_sentence_tensor(((kinds != 2) | ((positions >= 0) & (positions < depth))).all(),
                            'unary position is outside the recorded forest')
    live = kinds >= 0
    count = live.sum(1, keepdim=True)
    if check_capacity:
        require_sentence_tensor((count <= target_width).all(), 'recorded derivation exceeds its structural trace capacity')
    ordered = (~live).long().argsort(dim=1, stable=True)
    actions = torch.stack((kinds, local, args), -1)
    actions = torch.where(live[..., None], actions, -1).gather(1, ordered[..., None].expand(-1, -1, 3))
    columns = torch.where(live, slots, -1).gather(1, ordered)
    kinds, local = actions[..., 0], actions[..., 1]
    targets = torch.where(kinds == 0, stop, torch.where(kinds == 2, binary_count + local, local))
    reverse = count - 1 - torch.arange(target_width, device=active.device)
    targets = torch.where(reverse >= 0, targets.gather(1, reverse.clamp_min(0)), -1)
    return words, actions, targets, columns


def steps(reading):
    """The shared index for a native path, or the actions of an isolated journal."""
    index = getattr(reading, 'action_index', None)
    if index is not None:
        return index.steps
    if isinstance(reading, ReadingIndex):
        return reading.steps
    actions = reading if torch.is_tensor(reading) else reading.actions
    return actions.tolist()


class ReadingIndex:
    def __init__(self, actions, word_count, language, *, padded=False):
        self.steps = []
        self.operands = []
        self.parents = []
        self.word_actions = {}
        stack = []
        operations = actions.tolist()
        for at, (kind, local, word) in enumerate(operations):
            if padded and kind < 0:
                break
            self.parents.append(None)
            if kind == 0:
                if word_count is not None and (not 0 <= word < word_count or word in self.word_actions):
                    raise ValueError('a reading must shift each leaf exactly once')
                self.word_actions[word] = at
                local, children = -1, ()
                stack.append(at)
            elif kind == 3:
                if len(stack) < 2:
                    raise ValueError('transposition requires two operands')
                children = tuple(stack[-2:])
                stack[-2:] = reversed(children)
                word = -1
            else:
                if kind not in (1, 2):
                    raise ValueError('unknown sentence action')
                if padded:
                    rules = language._compose_binary_rules if kind == 1 else language._compose_unary_rules
                    if not 0 <= local < len(rules):
                        raise ValueError('invalid selected reference rule')
                if kind == 2 or getattr(language._compose_binary_rules[local], 'method_name', None) == 'bind':
                    offset = max(0, word) if kind == 2 else 0
                    position = len(stack) - 1 - offset
                    if position < 0:
                        raise ValueError('operation requires an occupied operand')
                    children = (stack[position],)
                    stack[position] = at
                    word = offset if kind == 2 else -1
                else:
                    if len(stack) < 2:
                        raise ValueError('binary operation requires two operands')
                    children = tuple(stack[-2:])
                    stack[-2:] = [at]
                    word = -1
                for child in children:
                    self.parents[child] = at
            self.operands.append(children)
            self.steps.append((kind, local, word))
        if word_count is not None and len(self.word_actions) != word_count:
            raise ValueError('a reading must shift each leaf exactly once')
        self.roots = tuple(stack)
        self.actions = actions.new_tensor(self.steps).reshape(-1, 3)

    @property
    def forest(self):
        """Nested display form, built only for a caller asking to inspect the tree."""
        nodes = []
        for at, (kind, local, word) in enumerate(self.steps):
            if kind == 0:
                nodes.append((kind, word))
            elif kind == 3:
                nodes.append(None)
            else:
                nodes.append((kind, local, at, *(nodes[child] for child in self.operands[at])))
        return tuple(nodes[at] for at in self.roots)


    def validate_interfaces(self, language):
        """The selected seat must still identify one declared execution interface."""
        from PartsOfSpeech import operation_name
        selected = {(kind, local) for kind, local, _ in self.steps if kind != 0}
        if not selected:
            return
        catalogs = {1: language._compose_binary_rules, 2: language._compose_unary_rules}
        matches = {}
        for kind, catalog in catalogs.items():
            for local, rule in enumerate(catalog):
                key = operation_name(rule), rule.canonical, rule.kernel_index
                matches.setdefault(key, []).append((kind, local))
        for kind, local in selected:
            rule = catalogs[1 if kind == 3 else kind][local]
            seat = language.parts_of_speech.seats[operation_name(rule)]
            key = language.parts_of_speech.seats[seat.concept].name, rule.canonical, rule.kernel_index
            if len(matches.get(key, ())) != 1:
                raise ValueError('the operator seat has no unique declared execution interface')
            if kind == 3 and len(rule.lhs.split(',')) != 2:
                raise ValueError('an unconsumed operator must be a declared transposition')
