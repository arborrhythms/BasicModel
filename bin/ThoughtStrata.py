"""Thought contents are views of the same short-term idea and pair stores."""
from dataclasses import replace
from copy import deepcopy

import torch

from Memory import STM0, OCCURRENCE, RELATION
from Meaning import ConceptualMeaning


class View:
    def __init__(self, store, root, generation):
        self.store, self.root, self.generation = store, root, generation

    def read(self):
        if self.store.generation != self.generation:
            raise ValueError('thought member reference has expired')
        store = self.store
        def members(ref):
            if ref[0] == OCCURRENCE:
                return [ref]
            left, right = store.parts.references(ref[1])
            return [*members(left), right]
        def meaning(ref):
            head, *contents = members(ref)
            metadata = dict(store.raw_meaning_of(store.index_of_row(head[1])).bindings)
            roles = store.slots.new_zeros(3, store.nDim)
            mask = torch.tensor(metadata['_roles'], dtype=torch.bool, device=roles.device)
            occupied = mask.nonzero().flatten().tolist()
            for role, member in zip(occupied, contents[:len(occupied)], strict=True):
                roles[role] = store.point_of_row(member[1])
            children = tuple(meaning(child) for child in contents[len(occupied):])
            def unquote(item):
                if isinstance(item, tuple):
                    if item[:1] == ('thought-child',): return ('constituent', *item[1:])
                    return tuple(unquote(part) for part in item)
                return item
            data = {name:unquote(item) for name,item in metadata['_metadata']}
            return ConceptualMeaning(roles, mask, **data, constituents=children)
        return meaning(self.root)

    def __getattr__(self, name):
        return getattr(self.read(), name)

    def detached(self):
        # The store owns detached content; trial snapshots retain its address.
        return self


def construct(batch, chunks, width):
    # A thought can retain one bounded set of constituent ideas. The recent
    # chunk count remains unchanged; these are its content and pair seats.
    return tuple(tuple(STM0(width, 4*chunks, percept_width=0,
        relation_capacity=4*chunks) for _ in range(chunks)) for _ in range(batch))


def write(store, meaning, *, key):
    nodes = []
    def visit(value):
        for child in value.constituents: visit(child)
        nodes.append(value)
    visit(meaning)
    count = sum(1+int(value.role_mask.sum()) for value in nodes)
    branches = count-1
    if count > store.capacity or branches >= store.parts.capacity:
        raise OverflowError('thought structure exceeds the constructed STM strata')
    store.expire()
    store.to(device=meaning.roles.device, dtype=meaning.roles.dtype)
    def append(point, bindings):
        value = replace(ConceptualMeaning.from_description(point.detach()), bindings=bindings)
        row = store.append_meaning(value, kind='unverified', when=False,
            document_key=('thought', tuple(float(value).hex() for value in key)),
            sentence_index=len(store))
        return OCCURRENCE, int(store.address_keys[row])
    def retain(value):
        children = [retain(child) for child in value.constituents]
        # Semantic metadata already uses local constituent indices. It owns
        # no tensor or topology; the ordered pairs below own the children.
        data = value.metadata()
        def quote(item):
            if isinstance(item, tuple):
                if item[:1] == ('constituent',): return ('thought-child', *item[1:])
                return tuple(quote(part) for part in item)
            return item
        metadata = {name:quote(item) for name,item in data.items()}
        head = append(value.roles[value.role_mask].amax(0),
            {'_roles': tuple(value.role_mask.tolist()), '_metadata': metadata})
        operands = [append(value.roles[role], {}) for role in value.role_mask.nonzero().flatten().tolist()]
        root = head
        for child in (*operands, *children):
            root = RELATION, store.parts.add(root, child, truth_plus=1.)
        return root
    return View(store, retain(meaning), store.generation)


def snapshot(stores):
    return tuple(tuple(None if not len(store) else (
        {name:value.clone() for name,value in store.state_dict().items()},
        {name:value.clone() for name,value in store.parts.state_dict().items()},
        {name:deepcopy(getattr(store, name)) for name in (
            '_semantic_rows', '_expectation_rows', '_texts', '_document_sources',
            '_cover_symbols', '_recurrence_identities', '_supplies')},
        store.generation) for store in row) for row in stores)


def restore(stores, saved):
    for row, snapshots in zip(stores, saved, strict=True):
        for store, state in zip(row, snapshots, strict=True):
            if state is None:
                if len(store): store.expire()
                continue
            tensors, parts, metadata, generation = state
            store.load_state_dict(tensors)
            store.parts.load_state_dict(parts)
            # This is a rollback of already admitted working content, not a
            # durable checkpoint import. An external occurrence may have
            # expired since the thought was written; restoring the thought
            # must neither resurrect it nor require it to be present again.
            for name, value in metadata.items():
                setattr(store, name, deepcopy(value))
            store.rebuild_addresses()
            store.generation = generation
