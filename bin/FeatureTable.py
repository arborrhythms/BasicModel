"""Device indices for deriving concept forms from their recorded basic percepts."""
import torch


class FeatureTable:
    def __init__(self):
        self.revision = None
        self.inventory_revision = None

    def inventory(self, alloc, basic, like):
        # Callers can type an existing symbol or attach a word to an existing
        # percept in place. Include the memberships themselves in the key.
        retained_ids = ({cid for cid, kinds in alloc.symbol_types.items() if 'concept' in kinds}
                        | set(alloc.individuals) | set(alloc.form_regions.values())) - alloc.retired
        records = tuple((row, key) for layer in alloc._layers.values()
                        for row, key in layer._tensor_row_keys.items())
        revision = (records, retained_ids, tuple(basic.words.items()), like.device)
        if revision != self.inventory_revision:
            retained, words = {}, {}
            for row, key in records:
                cid = key[1] if isinstance(key, tuple) else key
                retained[row] = cid in retained_ids
                words[row] = basic.words.get(cid, -1)
            self.retained = torch.zeros_like(like)
            self.fallback = torch.full_like(like, -1, dtype=torch.long)
            if records:
                rows = torch.tensor(list(retained), device=like.device)
                self.retained[rows] = like.new_tensor(list(retained.values()))
                self.fallback[rows] = rows.new_tensor(list(words.values()))
            self.inventory_revision = revision
        return self.retained, self.fallback

    def refresh(self, matrix, basic, device):
        revision = (id(matrix._index), matrix._next_edge_id, len(matrix._index),
                    id(basic), basic.revision, device)
        if revision == self.revision:
            return
        pairs = {}
        for (row, column), position in matrix._index.items():
            percept, pole = divmod(column, 2)
            pair = pairs.setdefault((row, percept), [-1, -1])
            pair[pole] = position
        keys = sorted(pairs)
        def indices(items):
            return torch.tensor(items, dtype=torch.long, device=device)
        self.rows = indices([row for row, _ in keys])
        self.percepts = indices([percept for _, percept in keys])
        self.positions = indices([pairs[key] for key in keys]).reshape(-1, 2)
        self.towers = []
        for side in (0, 1):
            percepts, members = [], []
            for percept, pair in enumerate(basic.occurrences[:len(basic.index)]):
                for member in pair[side]:
                    percepts.append(percept)
                    members.append(member)
            self.towers.append((indices(percepts), indices(members)))
        self.revision = revision

    def codes(self, basic, percepts):
        """Max of each percept's parts and min of its wholes, including empty wholes."""
        if not len(percepts):
            return basic.codes.new_empty(0, basic.width)
        if bool(((percepts < 0) | (percepts >= len(basic.index))).any()):
            raise ValueError('unavailable basic percept reference')
        lookup = percepts.new_full((len(basic.occurrences),), -1)
        lookup[percepts] = torch.arange(len(percepts), device=percepts.device)
        blocks = []
        for side, ((parents, members), width) in enumerate(zip(
                self.towers, (basic.identity_width, basic.property_width))):
            block = basic.codes.new_full((len(percepts), width), float(side))
            target = lookup[parents]
            selected = target >= 0
            if width and bool(selected.any()):
                refs = members[selected]
                if side and basic.whole is None and bool((refs >= -basic.identity_width).any()):
                    raise ValueError('basic percept property read requires its WholeSpace')
                codes = basic.whole_codes(refs) if side else basic.part_codes(refs)
                block.scatter_reduce_(0, target[selected, None].expand_as(codes), codes,
                                      reduce='amin' if side else 'amax', include_self=False)
            blocks.append(block)
        return torch.cat(blocks, -1)

    def forms(self, owner, matrix, rows, fallback):
        basic = owner.basic_percepts
        self.refresh(matrix, basic, rows.device)
        lookup = rows.new_full((owner.nVectors,), -1)
        lookup[rows] = torch.arange(len(rows), device=rows.device)
        target = lookup[self.rows]
        present = target >= 0
        positions = self.positions[present]
        # The old reader subtracted the two native weights before converting
        # the result to the percept chart. Preserve that order for mixed dtypes.
        dtype = torch.float32 if matrix.values is None else torch.promote_types(
            torch.float32, matrix.values.dtype)
        dtype = torch.promote_types(dtype, basic.codes.dtype)
        values = basic.codes.new_zeros(positions.shape, dtype=dtype)
        if len(positions):
            valid = positions >= 0
            values[valid] = matrix.values[positions[valid]].to(values)
        weights = values[:, 0] - values[:, 1]
        positive = weights > 0
        target, weights = target[present][positive], weights[positive].to(basic.codes)
        percepts = self.percepts[present][positive]
        counts = torch.bincount(target, minlength=len(rows))
        fallback = torch.where(counts == 0, fallback, -1)
        selected = fallback >= 0
        requested, inverse = torch.unique(torch.cat((percepts, fallback[selected])), return_inverse=True)
        codes = self.codes(basic, requested)
        forms = basic.codes.new_zeros(len(rows), basic.width)
        forms[selected] = codes[inverse[len(percepts):]]
        # Match the native sum's reduction length and order exactly. Padding
        # unequal lists or accumulating one float at a time can change the last bit.
        start = counts.cumsum(0) - counts
        for count in torch.unique(counts[counts > 0]).tolist():
            group = (counts == count).nonzero().flatten()
            take = start[group, None] + torch.arange(count, device=rows.device)
            weight = weights[take]
            forms[group] = (codes[inverse[take]] * weight[..., None]).sum(1) / weight.sum(1, keepdim=True)
        return forms
