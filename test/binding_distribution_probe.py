"""Read-only audit of the full conditional bind menu at initialization."""
from contextlib import contextmanager
from unittest.mock import patch
import torch


def retained_bank_distributions(model):
    """Audit the native chooser with guaranteed retained-context coverage.

    An untrained document reader need not construct an individual at all.
    Supply stored context here without training or forcing a chooser action;
    the ordinary document audit remains separate and unchanged.
    """
    from Layers import LTM0
    from ReferenceContext import ReferenceBank
    language = model.languageSpace
    layer = language.language_layer.operation_layer
    like = next(layer.parameters())
    width = layer.d_model
    records = []
    for order in (0, 2):  # Order-one binding belongs to the close-time resolver.
        for count in (1, 4, 16):
            # Distinct, nonnegative descriptions; no RNG draw or lexical hint.
            points = torch.arange(1, (count+2)*width+1, device=like.device,
                                  dtype=like.dtype).reshape(count+2, width)
            points = torch.nn.functional.normalize(points, dim=-1)
            store = LTM0(width, capacity=count)
            rows = [store.append_idea(point, order=order, document_key='retained-context',
                                     sentence_index=i) for i, point in enumerate(points[:count])]
            ids = store.row_ids[rows].to(device=like.device)[None]
            bank = ReferenceBank(ids, points[:count][None], torch.ones_like(ids, dtype=torch.bool),
                torch.zeros_like(ids, dtype=torch.bool), points[0][None],
                torch.ones(1, device=like.device, dtype=torch.bool), column_ids=ids.flatten(),
                interpreter=model._concept_owner().interpret,
                associations=ids.new_empty(0, 2), orders=torch.full_like(ids, order),
                close_references=True)
            buffer = points[-2:][None]
            identities = torch.arange(1, 3, device=like.device, dtype=torch.long)[None]
            state = (buffer, ids.new_tensor([2]), torch.full_like(identities, order),
                     torch.zeros_like(identities), identities, buffer.new_ones(1, 2))
            scope = torch.stack((torch.zeros_like(identities), identities), -1)
            with patch.object(language, '_reference_bank', bank, create=True), \
                    binding_distributions() as observed, torch.no_grad():
                language.choose_operation(state, torch.ones(1, device=like.device, dtype=torch.bool),
                    slots=1, sample=False, reference_scope=scope, policy_grad=False)
            for item in observed:
                item.update(context='controlled_retained_bank', stored_count=count, order=order)
            records.extend(observed)
    return records


@contextmanager
def binding_distributions():
    from Language import OperationSelectionLayer
    original = OperationSelectionLayer.forward
    records = []

    def observed(module, x, **kwargs):
        data = kwargs.get('reference_data')
        if not data or not data.get('binding_choices'):
            return original(module, x, **kwargs)
        # Observe the existing chooser before occupation/form priors,
        # reduction pressure and the expectation prime are added. Selection
        # still uses the complete ordinary logits; this is only a measurement.
        scored = {}
        def capture(label, function):
            def call(*args, **options):
                result = function(*args, **options)
                scored[label] = result[1].detach().cpu()
                return result
            return call
        with patch.object(module.chooser, 'score_binary',
                capture('binary', module.chooser.score_binary)), \
                patch.object(module.chooser, 'score_unary',
                capture('unary', module.chooser.score_unary)):
            result = original(module, x, **kwargs)
        complete = result[2]['logits'].detach().cpu()
        logits = torch.cat((scored['binary'].flatten(1), scored['unary'].flatten(1),
                            complete.new_zeros(len(x), 1)), -1)
        logits = logits.masked_fill(~torch.isfinite(complete), -torch.inf)
        selected = result[2]['action'].detach().cpu()
        nr, width = len(data['binary_ops']), x.shape[1]
        for label, positions, offset, arity in (
                ('binary', width-1, 0, 2), ('unary', width, (width-1)*nr, 1)):
            operations = data[label+'_ops']
            choices = data[label+'_choices'].detach().cpu().tolist()
            for row in range(len(x)):
                for position in range(positions):
                    groups = {}
                    for column, operation in enumerate(operations):
                        action = offset+position*len(operations)+column
                        if not bool(torch.isfinite(logits[row, action])):
                            continue
                        values = choices[row][position][column]
                        for role in range(arity):
                            if values[role] == -1:
                                continue  # this operand has no bind choice
                            key = operation, role, values[1-role]
                            group = groups.setdefault(key, {})
                            assert values[role] not in group, 'retained candidate duplicated in softmax'
                            group[values[role]] = action
                    for (operation, role, other), group in groups.items():
                        if not {0, -2}.issubset(group):
                            continue
                        alternatives = list(group)
                        actions = list(group.values())
                        probability = logits[row, actions].softmax(0)
                        records.append(dict(row=row, position=position, arity=arity,
                            operation=operation, role=role, other=other,
                            alternatives=alternatives, probabilities=probability.tolist(),
                            measured_term='chooser_content',
                            complete_probabilities=complete[row, actions].softmax(0).tolist(),
                            selected=int(selected[row]), actions=actions))
        return result

    with patch.object(OperationSelectionLayer, 'forward', observed):
        yield records
