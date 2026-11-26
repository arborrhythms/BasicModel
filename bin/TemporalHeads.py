"""Fixed-capacity publication of each stream's head at its own model tick."""
import torch


def publish(rows, ticks, heads):
    """Merge one head per stream, retaining the latest write at each coordinate.

    ``rows`` is sorted by tick, with all unused rows filled with -1. Storage
    and stream columns are reserved at the host boundary. Empty heads (-1 or
    0) leave previous writes intact. Ticks may differ between streams and may
    precede ticks already present in the index.
    """
    capacity, columns = rows.shape
    batch = len(heads)
    if not batch:
        return rows
    if columns < batch + 1:
        raise ValueError('sentence root index has no reserved stream columns')
    ticks = ticks.expand_as(heads)
    valid = (heads != -1) & (heads != 0)
    torch._assert_async((~valid | (ticks >= 0)).all(), 'a sentence root requires a nonnegative tick')
    own = torch.arange(columns - 1, device=rows.device)[None] == torch.arange(batch, device=rows.device)[:, None]
    incoming = torch.cat((ticks[:, None], torch.where(own & valid[:, None], heads[:, None], -1)), -1)
    incoming = torch.where(valid[:, None], incoming, -1)
    all_rows = torch.cat((rows, incoming))
    live = all_rows[:, 0] >= 0
    top = torch.iinfo(rows.dtype).max
    keys, order = torch.where(live, all_rows[:, 0], top).sort(stable=True)
    ordered = all_rows[order]
    live = keys != top
    first = live & torch.cat((live.new_ones(1), keys[1:] != keys[:-1]))
    groups = first.long().cumsum(0) - 1
    torch._assert_async((first.sum() <= capacity), 'sentence root index exceeds reserved tick capacity')
    target = torch.where(live, groups, capacity)
    source = torch.arange(len(all_rows), device=rows.device)[:, None].expand_as(ordered)
    occupied = live[:, None] & (ordered != -1)
    # For duplicate ticks, stable sorting puts this publication after the old
    # row. Select by write position, not by the signed value of the head.
    chosen = rows.new_full((capacity + 1, columns), -1).scatter_reduce(
        0, target[:, None].expand_as(ordered), torch.where(occupied, source, -1), 'amax')[:-1]
    values = ordered.gather(0, chosen.clamp_min(0))
    return torch.where(chosen >= 0, values, -1)
