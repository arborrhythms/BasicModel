"""Provisional content coverage shared by reference and field readers (§70)."""
import torch


def match(content, candidates, *, codebook=None, lane=None):
    """Return candidate-in-content and content-in-candidate coverage.

    A codebook with a form block compares only the following meaning
    coordinates. Without that layout the entire supplied point is content,
    including a percept's coordinates. ``lane`` is the field reader's
    independent for/against read: rows without meaning use their form.
    This function neither selects candidates nor detaches its operands.
    """
    def ratios(query, rows):
        width = min(query.shape[-1], rows.shape[-1])
        query, rows = query[..., :width], rows[..., :width]
        overlap = torch.minimum(query, rows).sum(-1)
        own, other = query.sum(-1), rows.sum(-1)
        return (torch.where(other > 0, overlap / torch.where(other > 0, other, 1.), 0.),
                torch.where(own > 0, overlap / torch.where(own > 0, own, 1.), 0.))

    derived = getattr(codebook, 'mereology', None)
    if lane is None:
        if derived is not None:
            # Keep the original reduction width, including narrowed rows.
            # Zero coordinates retain their addresses and reduction order.
            content = content * (torch.arange(content.shape[-1], device=content.device)
                                 >= derived.percept_event_width)
            candidates = candidates * (torch.arange(candidates.shape[-1], device=candidates.device)
                                       >= derived.percept_event_width)
        return ratios(content, candidates)
    if lane not in (0, 1):
        raise ValueError('a field match requires lane zero or one')
    width = min(content.shape[-1], candidates.shape[-1])
    form_width = width if derived is None else min(width, derived.percept_event_width)
    form = candidates[..., :form_width].clamp_min(0)
    query = content[..., :form_width]
    result = ratios((query if lane == 0 else -query).clamp_min(0), form)
    if form_width < width:
        if (width - form_width) % 2:
            raise ValueError('a field match requires two equal meaning blocks')
        lane_width = (width - form_width) // 2
        meaning = candidates[..., form_width:form_width + lane_width].clamp_min(0)
        start = form_width + lane * lane_width
        covered = ratios(content[..., start:start + lane_width].clamp_min(0), meaning)
        known = meaning.sum(-1) > 0
        result = tuple(torch.where(known, value, fallback)
                       for value, fallback in zip(covered, result))
    return result
