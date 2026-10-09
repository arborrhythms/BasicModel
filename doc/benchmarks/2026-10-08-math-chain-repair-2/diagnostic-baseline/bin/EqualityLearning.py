"""The conceptual equality residual of an asserted, selected reading.

Only actual equality operations in the current numerical journal contribute.
No lexical spelling, arithmetic label, reference order or desired program is
consulted. The loss follows the current operands back to their producing VP.
"""
import torch


def cost(language, programs, active, like):
    rules = language._compose_binary_rules
    terms, counts = [], []
    for row, program in enumerate(programs):
        local = []
        if program is not None and program.operation_values is not None and bool(active[row]):
            for index, (kind, operation, _word) in enumerate(program.actions.detach().cpu().tolist()):
                if kind != 1 or getattr(rules[operation], 'method_name', None) != 'equal':
                    continue
                left, right = program.operation_values[index, :2]
                # Equal means identity of the complete conceptual contents.
                # Both input sides are observed; the two faces retain their
                # normal pair result independently of this scalar loss term.
                local.append((left - right).square().mean())
        terms.append(torch.stack(local).mean() if local else like[row].sum() * 0)
        counts.append(len(local))
    return torch.stack(terms), counts
