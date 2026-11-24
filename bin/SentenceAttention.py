"""Sentence-local record of the words grammar has interpreted."""
import torch
from AttentionObjective import tolerate_heterogeneity


class SentenceField:
    def __init__(self, model, active, sentence_ids, sid):
        self.model = model
        self.scope = active & (sentence_ids == sid)
        self.positions = self.scope.any(0).nonzero().flatten()
        live = self.scope.index_select(1, self.positions)
        B, W = live.shape
        from PrependedSymbols import word_view
        from RowTime import subintervals
        bands = word_view(model.inputSpace, 'symbol_when')
        onset = model.when_encoding.decode_index(bands.index_select(1, self.positions))
        rank = live.long().cumsum(-1) - 1
        self.when = subintervals(onset, rank, live.sum(-1).clamp_min(1)[:, None])
        like = model.inputSpace._ar_embedded_N
        self.where = like.new_zeros(B, W, 2)
        self.support = like.new_zeros(B, W, 1, 2)
        self.identities = torch.full((B,W),-1,device=active.device,dtype=torch.long)
        self.admission = like.new_zeros(B,W)
        self.valid = torch.zeros_like(live)
        self.iterations = torch.zeros(B,device=active.device,dtype=torch.long)
        self.support_window = like.new_zeros(B, model.conceptualSpace.stm.capacity, W)
        self.reads = []
        self.parts = self.wholes = None

    def arrive(self, index, payload, native, rows):
        """Retain only the interpreted occurrence's own evidence and address."""
        B = self.scope.shape[0]
        column = (self.positions == index).long().argmax()
        identity = torch.where(payload[2] >= 0,payload[2],payload[4])
        registry = self.model.where_registry
        # A symbol has no address. An identified occurrence is placed where
        # it stands in the text, the address its word step was given.
        from PrependedSymbols import word_view
        band = word_view(self.model.inputSpace, 'symbol_where')[:, index]
        start = registry.encoding.decode_index(band).to(torch.long)
        known = torch.where((identity >= 0)[:, None] & band.ne(0).any(-1, keepdim=True),
                            torch.stack((start, start + 1), -1), -1)
        pair = self.model._attention_words.poles[:,index]
        if native:
            parts, wholes, codes, roles, evidence, mask = native
            for name, carrier in (('parts', parts), ('wholes', wholes)):
                value = torch.where(rows[:, None, None], carrier[:, -1], 0.)
                prior = getattr(self, name)
                setattr(self, name, value if prior is None else prior + value)
            address = codes + torch.where(roles == 0,registry.slices['parts'][0],
                int(self.model.wholeSpace.subspace.what.where_offset))
            address = address[:,None].expand_as(evidence)
            mask = mask & (codes[:,None]>=0) & rows[:,None,None]
            lanes = torch.stack((evidence.clamp_min(0),(-evidence).clamp_min(0)),-1)
            lanes = torch.where(mask[...,None],lanes,0.)
            low = torch.where(mask,address,registry.capacity).flatten(1).amin(-1)
            high = torch.where(mask,address+1,0).flatten(1).amax(-1)
            extent = torch.where((identity>=0)[:,None],known,torch.stack((low,high),-1))
            support = torch.cat((lanes.flatten(1,2),pair[:,None]),1)
        else:
            extent,support = known,pair[:,None]
        support = tolerate_heterogeneity(support,self.model.het_tolerance)
        if self.support.shape[2] != support.shape[1]:
            if self.reads:
                raise ValueError('an arrival changed the native support width')
            self.support = support.new_zeros(B,len(self.positions),support.shape[1],2)
        at = (torch.arange(len(self.positions),device=rows.device)==column)[None]&rows[:,None]
        self.support = torch.where(at[...,None,None],support[:,None],self.support)
        self.where = torch.where(at[...,None],extent[:,None].to(self.where),self.where)
        self.identities = torch.where(at,identity[:,None],self.identities)
        self.valid |= at
        self.admission = torch.where(at,1.,self.admission)
        self.iterations += rows.long()
        self.reads.append(dict(index=index,rows=rows.clone()))

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
        from Language import TheGrammar, GRAMMAR_LAYER_CLASSES
        pairs = torch.tensor([bool(getattr(GRAMMAR_LAYER_CLASSES.get(rule.method_name), 'preserves_pair', False))
                              for rule in TheGrammar.rules], device=rows.device)
        for column in columns:
            at = (torch.full((B, 1), column, device=rows.device, dtype=torch.long)
                  if isinstance(column, int) else column[:, None])
            rule_ids = language[1].gather(1, at).squeeze(1)
            pair = pairs[rule_ids.clamp(0,len(pairs)-1)] if pairs.numel() else torch.zeros_like(rows)
            binary = rows & language[3].gather(1, at).squeeze(1) & (language[2].gather(1, at).squeeze(1) == 2)
            transpose = binary & pair
            binary &= ~pair
            if K > 1:
                merged = torch.maximum(window[:, 0], window[:, 1])
                folded = torch.cat((merged[:, None], window[:, 2:], torch.zeros_like(window[:, :1])), 1)
                window = torch.where(binary[:, None, None], folded, window)
                swapped = torch.cat((window[:,1:2], window[:,:1], window[:,2:]),1)
                window = torch.where(transpose[:,None,None], swapped, window)
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
            if live.shape[-1] == 0:
                # A presentation with no word (a scalar alone) still closes.
                # No slot holds a word's witness, so nothing is placed.
                result[name] = bands.new_zeros(*occupied.shape, 2)
                continue
            mask = live.to(bands.device)
            low = torch.where(mask, bands[:, None, :, 0], torch.inf).amin(-1)
            high = torch.where(mask, bands[:, None, :, 1], -torch.inf).amax(-1)
            result[name] = torch.where(occupied[..., None].to(bands.device), torch.stack((low, high), -1), 0.)
        result['lanes'] = torch.einsum('bkw,bwf->bkf', self.support_window, self.support.sum(-2))
        return result

    def source_positions(self, row, *, valid=None):
        """Source columns of the accepted arrivals in this stream's read order."""
        if not self.reads:
            return self.positions.new_empty(0)
        columns = torch.stack([entry['index'] for entry in self.reads])
        accepted = torch.stack([entry['rows'][row] for entry in self.reads])
        if valid is not None:
            accepted = accepted & valid[columns]
        return columns[accepted]

    def source_order(self, row):
        return self.source_positions(row).tolist()

    def full_admission(self):
        return self.support.new_zeros(self.scope.shape).scatter(1,
            self.positions[None].expand(len(self.support), -1), self.admission)

    def grammar_leaves(self, prior):
        """Reconstruct exactly the leaves this trial placed into compose.

        Attention can admit a leaf absent before the trial's hand-off. The
        old mask belongs only to the other sentences, not this trial's tree.
        """
        return torch.where(self.scope, self.full_admission().detach().bool(), prior)

    def ordered_reconstruction(self, recovered):
        """Compact the already inverted supports in their actual read order."""
        compact = torch.zeros_like(recovered)
        for b in range(len(recovered)):
            order = self.source_order(b)
            if order:
                compact[b, :len(order)] = recovered[b, order]
        return compact, self.iterations

    def report(self):
        return dict(iterations=self.iterations.detach(),admission=self.admission.detach(),
            eligible=self.valid.detach(),where=self.where.detach(),when=self.when.detach(),
            support=self.support.detach(),positions=self.positions.detach(),identities=self.identities.detach(),
            reads=[{key:value.detach() for key,value in entry.items()} for entry in self.reads],
            slot_support=self.support_window.detach(),
            slot_stamps={key:value.detach() for key,value in self.placements().items()})


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
