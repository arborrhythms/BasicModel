"""Device indices for deriving concept forms from their recorded basic percepts
(the allocator's record of order-zero members)."""
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

    def refresh(self, record, basic, device):
        revision = (id(record), record.revision, id(basic), basic.revision, device)
        if revision == self.revision:
            return
        pairs = {}
        for (row, percept, negated), weight in record.items():
            pair = pairs.setdefault((row, percept), [0., 0.])
            pair[int(negated)] = weight
        keys = sorted(pairs)
        def indices(items):
            return torch.tensor(items, dtype=torch.long, device=device)
        self.rows = indices([row for row, _ in keys])
        self.percepts = indices([percept for _, percept in keys])
        self.weights = torch.tensor([pairs[key] for key in keys], device=device).reshape(-1, 2)
        self.towers = []
        for side in (0, 1):
            percepts, members = [], []
            for percept, pair in enumerate(basic.occurrences[:len(basic.index)]):
                for member in pair[side]:
                    percepts.append(percept)
                    members.append(member)
            self.towers.append((indices(percepts), indices(members)))
        # Place definitions change when this inventory changes. Transfer their
        # kept columns together here, then gather only the requested rows.
        place_rows = sorted(basic.place_rows)
        self.place_ids = indices(place_rows)
        if place_rows:
            values = torch.zeros(len(place_rows), basic.width, device='cpu', dtype=torch.float32)
            for seat, row in enumerate(place_rows):
                what = basic.place_rows[row][2]
                if what is not None:
                    values[seat] = what
            self.place_values = values.to(basic.value_codes)
        else:
            self.place_values = None
        self.revision = revision

    def codes(self, basic, percepts, band=0):
        """Max of each percept's parts and min of its wholes, including empty
        wholes; a place's, the what its row keeps (nothing where it keeps
        none). With ``band``, each row's where and when entries appended
        (``BasicPercepts.place_code``)."""
        if not len(percepts):
            return basic.value_codes.new_empty(0, basic.width + int(band))
        from Meaning import require_sentence_tensor
        require_sentence_tensor(((percepts >= 0) & (percepts < len(basic.index))).all(),
                                'unavailable basic percept reference')
        lookup = percepts.new_full((len(basic.occurrences),), -1)
        lookup[percepts] = torch.arange(len(percepts), device=percepts.device)
        blocks = []
        for side, ((parents, members), width) in enumerate(zip(
                self.towers, (basic.identity_width, basic.property_width))):
            block = basic.value_codes.new_full((len(percepts), width), float(side))
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
        result = torch.cat(blocks, -1)
        # A place's row has no parts and no wholes to be read from: its what
        # is the what it keeps.
        if self.place_values is not None:
            seats = torch.searchsorted(self.place_ids, percepts.contiguous()).clamp_max(len(self.place_ids) - 1)
            result = torch.where((self.place_ids[seats] == percepts)[:, None], self.place_values[seats], result)
        if band:
            result = torch.cat((result, basic.place_code(percepts, band).to(result)), -1)
        return result

    def forms(self, owner, record, rows, fallback, band=0):
        basic = owner.basic_percepts
        self.refresh(record, basic, rows.device)
        lookup = rows.new_full((owner.nVectors,), -1)
        lookup[rows] = torch.arange(len(rows), device=rows.device)
        target = lookup[self.rows]
        present = target >= 0
        # A member's weight less its weight as a Not member, then the
        # percept chart.
        values = self.weights[present].to(torch.promote_types(torch.float32, basic.value_codes.dtype))
        weights = values[:, 0] - values[:, 1]
        positive = weights > 0
        target, weights = target[present][positive], weights[positive].to(basic.value_codes)
        percepts = self.percepts[present][positive]
        counts = torch.bincount(target, minlength=len(rows))
        fallback = torch.where(counts == 0, fallback, -1)
        selected = fallback >= 0
        requested, inverse = torch.unique(torch.cat((percepts, fallback[selected])), return_inverse=True)
        codes = self.codes(basic, requested, band)
        forms = basic.value_codes.new_zeros(len(rows), basic.width + int(band))
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
