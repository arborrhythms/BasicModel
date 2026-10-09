"""Read-only audit of the full conditional bind menu at initialization."""
from contextlib import contextmanager
from unittest.mock import patch
import torch


@contextmanager
def binding_distributions():
    from Language import OperationSelectionLayer
    original = OperationSelectionLayer.forward
    records = []

    def observed(module, x, **kwargs):
        result = original(module, x, **kwargs)
        data = kwargs.get('reference_data')
        if not data or not data.get('binding_choices'):
            return result
        logits = result[2]['logits'].detach().cpu()
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
                            selected=int(selected[row]), actions=actions))
        return result

    with patch.object(OperationSelectionLayer, 'forward', observed):
        yield records
