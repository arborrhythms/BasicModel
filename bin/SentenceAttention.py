"""Sentence-local candidates: native word wholes and their paired evidence."""
import torch
from AttentionTraversal import FieldTraversal


class SentenceField(FieldTraversal):
    def __init__(self, model, cache, active, sentence_ids, sid, allowance, observed):
        self.model, self.observed = model, observed.detach()
        self.scope = active & (sentence_ids == sid)
        self.positions = torch.stack([entry[0] for entry in cache]).long()
        valid = self.scope.index_select(1, self.positions)
        B, W = valid.shape
        registry = model.where_registry
        native = [entry[2] for entry in cache]
        self.parts = torch.stack([entry[0][:, -1] for entry in native]).sum(0)
        self.wholes = torch.stack([entry[1][:, -1] for entry in native]).sum(0)
        bands = model.inputSpace._ar_word_symbol_when
        onset = model.when_encoding.decode_index(bands.index_select(1, self.positions))
        rank = valid.long().cumsum(-1) - 1
        count = valid.sum(-1).clamp_min(1)[:, None]
        when = torch.stack((onset + rank / count, onset + (rank + 1) / count), -1)
        addresses, features, masks, spans, keys, identities = [], [], [], [], [], []
        reading = getattr(model, '_attention_words', None)
        for column, (index, payload, entry) in enumerate(cache):
            _, _, codes, roles, evidence, mask = entry
            live = valid[:, column] & payload[8].reshape(B)
            if reading is not None:
                live &= reading.accepted[:, index]
            valid[:, column] = live
            address = codes + torch.where(roles == 0, registry.slices['parts'][0],
                                         int(model.wholeSpace.subspace.what.where_offset))
            address = address[:, None].expand_as(evidence)
            mask = mask & (codes[:, None] >= 0) & live[:, None, None]
            lanes = torch.stack((evidence.clamp_min(0), (-evidence).clamp_min(0)), -1)
            lanes = torch.where(mask[..., None], lanes, 0.)
            identity = torch.where(payload[2] >= 0, payload[2], payload[4])
            identities.append(identity)
            known = registry.intervals('symbols', identity)
            lo = torch.where(mask, address, registry.capacity).flatten(1).amin(-1)
            hi = torch.where(mask, address + 1, 0).flatten(1).amax(-1)
            extent = torch.where((identity >= 0)[:, None], known, torch.stack((lo, hi), -1))
            spans.append(extent)
            pair = payload[11] if reading is None else reading.poles[:, index]
            pair = torch.where(live[:, None], pair.to(lanes), 0.)
            addresses.append(torch.cat((address.flatten(1), known[:, :1]), 1))
            features.append(torch.cat((lanes.flatten(1, 2), pair[:, None]), 1))
            masks.append(torch.cat((mask.flatten(1), (live & (identity >= 0))[:, None]), 1))
            keys.append(payload[0].mean(1))
        self.addresses = torch.stack(addresses, 1)
        self.masks = torch.stack(masks, 1)
        self.keys = torch.stack(keys, 1)
        self.identities = torch.stack(identities, 1)
        support = torch.stack(features, 1)
        super().__init__(support, torch.stack(spans, 1).to(support), when.to(support),
                         valid, allowance, tolerance=model.het_tolerance)
        self.support_window = support.new_zeros(B, model.conceptualSpace.stm.capacity, W)
        self._closing = torch.zeros(B, dtype=torch.bool, device=active.device)

    def remember(self, index, rows, language, *, closing=False):
        """Provenance of the existing STM slots, without a second content store.

        Grammar's recorded operations move these source supports with their
        slots. A composed whole retains the witnesses of both operands.
        Native STM remains the sole owner of the composed numerical value.
        """
        window = self.support_window
        B, K, W = window.shape
        if not closing:
            column = (self.positions == index).long().argmax()
            head = torch.nn.functional.one_hot(column, W).to(window)[None].expand(B, -1)
            window = torch.where(rows[:, None, None], torch.cat((head[:, None], window[:, :-1]), 1), window)
            columns = range(3 * int(index), 3 * int(index) + 3)
        else:
            width = self.scope.shape[1]
            group = int(index) + 1 if int(index) < width - 1 else 0
            # The final sentence uses group zero even in a padded bucket.
            all_last = torch.where(self.model.inputSpace._word_active_mask,
                torch.arange(width, device=rows.device)[None], -1).amax(-1)
            groups = torch.where(all_last == index, 0, group)
            columns = [3 * width + groups * (2 * K) + r for r in range(2 * K)]
        for column in columns:
            at = (torch.full((B, 1), column, device=rows.device, dtype=torch.long)
                  if isinstance(column, int) else column[:, None])
            binary = rows & language[6].gather(1, at).squeeze(1) & (language[5].gather(1, at).squeeze(1) == 2)
            if K > 1:
                merged = torch.maximum(window[:, 0], window[:, 1])
                folded = torch.cat((merged[:, None], window[:, 2:], torch.zeros_like(window[:, :1])), 1)
                window = torch.where(binary[:, None, None], folded, window)
        self.support_window = window

    def placements(self):
        """Read the stamps and lanes carried by the occupied STM slots.

        The source mask indexes the immutable candidate support bank. Binary
        composition joins witnesses; a later read never re-places them.
        """
        live = self.support_window.bool()
        occupied = live.any(-1)
        result = {}
        for name, bands in (('where', self.where), ('when', self.when)):
            low = torch.where(live, bands[:, None, :, 0], torch.inf).amin(-1)
            high = torch.where(live, bands[:, None, :, 1], -torch.inf).amax(-1)
            result[name] = torch.where(occupied[..., None], torch.stack((low, high), -1), 0.)
        result['lanes'] = torch.einsum('bkw,bwf->bkf', self.support_window, self.support.sum(-2))
        return result

    def place(self, stm):
        from ModelCandidateAttention import context
        layer = self.model.candidate_attention
        if layer is None:
            return None
        B, W, K, _ = self.support.shape
        valid = self.masks & self.remaining[..., None]
        when = self.when[:, :, None].expand(-1, -1, K, -1).reshape(B, W * K, 2)
        inputs = context(self.model, self.support.reshape(B, W * K, 2),
            self.addresses.reshape(B, W * K), valid.reshape(B, W * K), when=when,
            parts=self.parts, wholes=self.wholes, stm=stm, placements=self.placements())
        # Native key, structural stamps, both support magnitudes, unreadness,
        # and priming salience; no desired text or target position enters.
        heat = self.model._concept_owner().priming_weights(batch=B)
        ids = self.identities.clamp(0, heat.shape[1] - 1)
        salience = torch.where(self.identities >= 0, heat.gather(1, ids), 1.)
        # Express temporal relations in this field's units. Dividing a
        # fraction of a sentence by the entire timestamp range buried this
        # input below the content keys. This is a coordinate chart, not a
        # preferred direction or a selection rule.
        onset = torch.where(self.valid, self.when[..., 0], torch.inf).amin(-1, keepdim=True)
        end = torch.where(self.valid, self.when[..., 1], -torch.inf).amax(-1, keepdim=True)
        onset = torch.where(self.valid.any(-1, keepdim=True), onset, 0.)
        duration = torch.where(self.valid.any(-1, keepdim=True), end-onset, 1.).clamp_min(1e-6)
        relative_when = (self.when-onset[..., None]) / duration[..., None]
        structure = torch.cat((self.where / self.model.where_registry.capacity,
            relative_when,
            self.support.sum(-2), self.remaining[..., None].to(self.keys),
            salience[..., None]), -1)
        return layer(inputs, torch.cat((self.keys, structure), -1))

    def source_order(self, row):
        return [int(self.positions[entry['action'][row]]) for entry in self.reads
                if bool(entry['admitted'][row].any())]

    def full_admission(self):
        return self.support.new_zeros(self.scope.shape).scatter(1,
            self.positions[None].expand(len(self.support), -1), self.admission)

    def costs(self, record, recovered, *, defined, step_cost):
        from AttentionObjective import field_cost
        admission = self.full_admission()
        B, W, D = record.word_values.shape
        admitted = self.scope & admission.bool()
        if recovered.shape != record.word_values.shape:
            raise ValueError('derivation reconstruction must align with the admitted field')
        recovered = torch.where(admitted[..., None], recovered, 0.)
        for entry in self.reads:
            live = entry['admitted'].any(-1)
            index = self.positions[entry['action'].clamp_max(len(self.positions)-1)]
            value = recovered.gather(1, index[:, None, None].expand(B, 1, D)).squeeze(1)
            value = torch.where(live[:, None], value, 0.)
            target = self.observed.gather(1, index[:, None, None].expand(B, 1, D)).squeeze(1)
            coordinates = defined.gather(1, index[:, None, None].expand(B, 1, D)).squeeze(1)
            entry['decoded'] = value.detach()
            entry['defined_coordinates'] = coordinates & live[:, None]
            entry['unrecovered_words'] = (live & ~coordinates.all(-1)).long()
            entry['unrecovered_coordinates'] = (live[:, None] & ~coordinates).sum(-1)
            error = torch.where(coordinates, target-value, 0.).abs().sum(-1)
            entry['inside'] = torch.where(live, error, 0.).detach()
            entry['work'] = live.to(value) * step_cost
        costs = field_cost(self.observed, recovered, admission, self.scope,
            floor=self.model.attention_floor, defined=defined,
            iterations=self.iterations, step_cost=step_cost)
        residual = self.observed.abs().sum(-1)
        unread = self.scope.clone()
        for entry in self.reads:
            read = torch.zeros_like(unread).scatter(1,
                self.positions[None].expand(B, -1), entry['admitted'])
            unread &= ~read
            entry['outside'] = (self.model.attention_floor * residual * unread).sum(-1).detach()
        return costs, recovered

    def lesson_cost(self):
        terms = [read['lesson_cross_entropy'] for read in self.reads
                 if 'lesson_cross_entropy' in read]
        return (torch.stack(terms).sum(0) / self.iterations.clamp_min(1)
                if terms else None)

    def ordered_reconstruction(self, recovered):
        """Compact the already inverted supports in their actual read order."""
        compact = torch.zeros_like(recovered)
        for b in range(len(recovered)):
            order = self.source_order(b)
            if order:
                compact[b, :len(order)] = recovered[b, order]
        return compact, self.iterations

    def report(self):
        result = super().report()
        result['positions'], result['identities'] = self.positions.detach(), self.identities.detach()
        result['slot_support'] = self.support_window.detach()
        result['slot_stamps'] = {key: value.detach() for key, value in self.placements().items()}
        return result


def masked_payload(payload, rows, *, closing=False):
    values = list(payload)
    values[7] = values[7] & rows[:, None]
    values[8] = values[8] & rows[:, None] & (not closing)
    if closing:
        for index in (0, 3, 6, 9, 10, 11):
            values[index] = torch.zeros_like(values[index])
        for index in (2, 4):
            values[index] = torch.full_like(values[index], -1)
    return tuple(values)
