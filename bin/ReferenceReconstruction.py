"""Reconstruct a closing mention's description from its chosen record."""
import torch


def cost(resolver, plans, like):
    """Sum uncovered description shares per sentence, without reading memory.

    The preview already owns the candidate records. A first mention and an
    elided role supply no reference reconstruction; arrivals are separate
    thought operations. Same-close aliases are ordinary selected records.
    """
    if len(plans) != len(like):
        raise ValueError('reference reconstruction plans must align with sentence rows')
    rows = []
    for plan in plans:
        total = like.new_zeros(())
        for item in (() if plan is None else plan['items']):
            selected = item['selected']
            if (selected is None or item['mode'] == 'mint' or item.get('own_object') or item.get('compose_dummy')
                    or not resolver.has_word(item['node'])):
                continue
            query = item['content']
            mass = query.sum()
            if not bool(mass > 0):
                continue
            record = item.get('record', plan['individuals'].get(selected))
            if record is None:
                raise RuntimeError('a selected bind has no scored record')
            candidate = resolver.coordinates(record['value'] if 'value' in record else record['content'])
            width = min(query.shape[-1], candidate.shape[-1])
            covered = torch.minimum(query[..., :width], candidate[..., :width]).sum()
            total = total + 1 - covered / mass
        rows.append(total)
    return torch.stack(rows) if rows else like.new_zeros(0)
