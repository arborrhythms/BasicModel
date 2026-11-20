"""Supply a fixture's observed field through the native placement reader."""
import torch


@torch.no_grad()
def place_types(owner, identities=None):
    """Place types while keeping the fixture's explicit word coordinates fixed."""
    bank = owner.similarity_codebook
    codes = getattr(bank, 'mereology', None)
    if codes is None:
        return
    if identities is None:
        identities = [cid for cid in owner._concept_allocator.concepts
                      if owner._concept_row_of(cid) is not None
                      and owner._concept_allocator.order_of(cid) == 0]
    owner._observed_knowing.zero_()
    rows = [owner._concept_row_of(cid) for cid in identities]
    owner._observed_knowing[0, rows, 0] = 1.
    context = owner.placement_context(0)
    active = torch.tensor([row for row in (context > 0).nonzero().flatten().tolist()
                           if owner._row_order(row) > 0], device=bank.W.device, dtype=torch.long)
    codes.place_rows(active.tolist(), context, b'observed-placement-fixture')
    bank.lookup_rows(active)


@torch.no_grad()
def place_at(owner, identity, coordinates):
    """Declare a geometry fixture; membership no longer implies a centroid."""
    bank = owner.similarity_codebook
    row = owner._concept_row_of(identity)
    value = bank.W.new_tensor(coordinates)
    bank.mereology.placement[row] = torch.nn.functional.normalize(value, dim=0)
    bank.mereology.placement_witnesses[row] = 1
    bank.mereology.placement_dirty[row] = True
    bank.lookup_rows([row])
