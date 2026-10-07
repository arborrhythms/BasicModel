"""Count every declared pole consumer without changing values or drawing RNG."""
from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def observe_handoff(result):
    import ModelAttention
    original = ModelAttention.read_poles
    result['pole_consumers'] = dict.fromkeys(ModelAttention.POLE_CONSUMERS, 0)
    result['containment'] = {}
    def observed(model, consumer):
        pair = original(model, consumer)
        if pair is not None:
            result['pole_consumers'][consumer] += 1
        return pair
    from Models import BasicModel
    forward = BasicModel.forward
    def observed_forward(model, *args, **kwargs):
        value = forward(model, *args, **kwargs)
        audits = [cs.similarity_codebook.mereology.containment_audit()
                  for cs in model.conceptualSpaces
                  if getattr(cs.similarity_codebook, 'mereology', None) is not None]
        result['containment'].setdefault('start', audits)
        result['containment']['end'] = audits
        result['containment']['largest'] = max(result['containment'].get('largest',0.),
            max((r[phase]['largest'] for r in audits for phase in ('before','after')),default=0.))
        return value
    with patch.object(ModelAttention, 'read_poles', observed), patch.object(BasicModel, 'forward', observed_forward):
        yield result
